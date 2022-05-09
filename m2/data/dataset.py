import os
import numpy as np
import itertools
import collections
from .example import Example
from .utils import nostdout
from pycocotools.coco import COCO as pyCOCO


class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = fields

    def collate_fn(self):
        def collate(batch):
            tensors = {}
            for field_name, field in self.fields.items():
                data = [x[field_name] for x in batch]
                tensors[field_name] = field.process(data)

            return tensors

        return collate

    def __getitem__(self, i):
        example = self.examples[i]
        data = {}
        for field_name, field in self.fields.items():
            data[field_name] = field.preprocess(getattr(example, field_name))

        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ValueDataset(Dataset):
    def __init__(self, examples, fields, dictionary):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fn()(value_batch_flattened)

            value_tensors = {}
            lengths = [0, ] + list(itertools.accumulate([len(x) for x in batch]))
            for k, v in value_tensors_flattened.items():
                value_tensors[k] = [v[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]

            return value_tensors
        
        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(Dataset):
    def __init__(self, examples, fields, key_fields, aux_fields):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        for field in key_fields:
            assert (field in fields)
        
        if not isinstance(aux_fields, (tuple, list)):
            aux_fields = (aux_fields,)
        for field in aux_fields:
            assert (field in fields) and (field not in key_fields)
        
        key_fields = {k: fields[k] for k in key_fields}
        aux_fields = {k: fields[k] for k in aux_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields and k not in aux_fields}

        dictionary = collections.defaultdict(list)
        key_examples = []
        aux_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            aux_example = Example.fromdict({k: getattr(e, k) for k in aux_fields})
            value_example = Example.fromdict({k: getattr(e, k) for k in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)
                aux_examples.append(aux_example)

            value_examples.append(value_example)
            dictionary[key_dict[key_example]].append(i)

        self.key_dataset = Dataset(key_examples, key_fields)
        self.aux_dataset = Dataset(aux_examples, aux_fields)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch, axu_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)
            value_tensors = self.value_dataset.collate_fn()(value_batch)
            aux_tensors = self.aux_dataset.collate_fn()(axu_batch)

            return {**key_tensors, **value_tensors, **aux_tensors}

        return collate

    def __getitem__(self, i):
        return (self.key_dataset[i], self.value_dataset[i], self.aux_dataset[i])

    def __len__(self):
        return len(self.key_dataset)


def unique(sequence):
    seen = set()
    if isinstance(sequence[0], list):
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
    else:
        return [x for x in sequence if not (x in seen or seen.add(x))]


class PairedDataset(Dataset):
    def __init__(self, examples, fields):
        super(PairedDataset, self).__init__(examples, fields)
        self.img_id_field = self.fields["img_id"]
        self.object_field = self.fields['object']
        self.text_field = self.fields['text']
        self.txt_ctx_field = self.fields["txt_ctx"]
        self.vis_ctx_field = self.fields["vis_ctx"]
        
    def image_set(self):
        img_list = [e.object for e in self.examples]
        image_set = unique(img_list)
        examples = [Example.fromdict({'object': i}) for i in image_set]
        dataset = Dataset(examples, {'object': self.object_field})
        return dataset

    def text_set(self):
        text_list = [e.text for e in self.examples]
        text_list = unique(text_list)
        examples = [Example.fromdict({'text': t}) for t in text_list]
        dataset = Dataset(examples, {'text': self.text_field})
        return dataset

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        
        aux_fields = ["img_id", "txt_ctx", "vis_ctx"]
        dataset = DictionaryDataset(
            self.examples, fields, key_fields='object', aux_fields=aux_fields
        )
        return dataset

    def text_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        
        aux_fields = ["img_id", "txt_ctx", "vis_ctx"]
        dataset = DictionaryDataset(
            self.examples, fields, key_fields='text', aux_fields=aux_fields
        )
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


class COCO(PairedDataset):
    def __init__(
        self, fields, ann_root, id_root=None, use_restval=True, cut_validation=False
    ):
        roots = {}
        roots['train'] = os.path.join(ann_root, 'captions_train2014.json')
        roots['val'] = os.path.join(ann_root, 'captions_val2014.json')
        roots['test'] = os.path.join(ann_root, 'captions_val2014.json')
        roots['trainrestval'] = (roots['train'], roots['val'])
        
        if id_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(id_root, 'coco_train_ids.npy'))
            ids['val'] = np.load(os.path.join(id_root, 'coco_dev_ids.npy'))
            if cut_validation:
                ids['val'] = ids['val'][:5000]
            ids['test'] = np.load(os.path.join(id_root, 'coco_test_ids.npy'))
            ids['trainrestval'] = (
                ids['train'],
                np.load(os.path.join(id_root, 'coco_restval_ids.npy')))

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']
        else:
            ids = None

        with nostdout():
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(roots, ids)
        examples = self.train_examples + self.val_examples + self.test_examples
        
        super(COCO, self).__init__(examples, fields)

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    def get_samples(self, roots, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []

        for split in ['train', 'val', 'test']:
            if isinstance(roots[split], tuple):
                coco_dataset = (pyCOCO(roots[split][0]), pyCOCO(roots[split][1]))
            else:
                coco_dataset = (pyCOCO(roots[split]),)

            if ids_dataset is None:
                ids = list(coco_dataset.anns.keys())
            else:
                ids = ids_dataset[split]
            
            if isinstance(ids, tuple):
                bp = len(ids[0])
                ids = list(ids[0]) + list(ids[1])
            else:
                bp = len(ids)

            for index in range(len(ids)):
                if index < bp:
                    coco = coco_dataset[0]
                else:
                    coco = coco_dataset[1]

                ann_id = ids[index]
                caption = coco.anns[ann_id]['caption']
                img_id = coco.anns[ann_id]['image_id']
                filename = coco.loadImgs(img_id)[0]['file_name']
                filename = f"{filename.split('_')[1]}/{filename}"

                example = {
                    "img_id": img_id,
                    "object": img_id,
                    "text": caption,
                    "txt_ctx": img_id,
                    "vis_ctx": img_id
                }
                example = Example.fromdict(example)

                if split == 'train':
                    train_samples.append(example)
                elif split == 'val':
                    val_samples.append(example)
                elif split == 'test':
                    test_samples.append(example)

        return train_samples, val_samples, test_samples


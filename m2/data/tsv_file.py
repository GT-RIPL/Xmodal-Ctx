# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

import logging
import os
import os.path as op
import argparse
from pathlib import Path
import json
import numpy as np
import base64
import h5py
from tqdm import tqdm
import shutil


def generate_lineidx_file(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)


class TSVFile(object):
    def __init__(self, tsv_file, generate_lineidx=False):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None
        # the process always keeps the process which opens the file. 
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and generate_lineidx:
            generate_lineidx_file(self.tsv_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            logging.info('{}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logging.info('loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.info('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert tsv features to hdf5')
    parser.add_argument('--features_dir', type=str)
    parser.add_argument('--save_file', type=str)
    args = parser.parse_args()
    args.features_dir = Path(args.features_dir)

    args.features_dir.mkdir(parents=True, exist_ok=True)
    args.save_file = Path(args.save_file)
    if args.save_file.exists():
        args.save_file.unlink()

    f = h5py.File(args.save_file, "w")
    for split in ["train", "val", "test"]:
        feat_tsv = TSVFile(str(args.features_dir/f"{split}.feature.tsv"))
        label_tsv = TSVFile(str(args.features_dir/f"{split}.label.tsv"))
        ids2index = {label_tsv.seek(i)[0]: i for i in range(label_tsv.num_rows())}
        
        print(f"Start parsing the {split} set.")
        for img_id in tqdm(ids2index.keys(), dynamic_ncols=True):
            idx = ids2index[str(img_id)]
            feat_info = json.loads(feat_tsv.seek(idx)[1])
            num_boxes = feat_info['num_boxes']
            features = np.frombuffer(base64.b64decode(feat_info['features']), np.float32)
            features = np.copy(features.reshape((num_boxes, -1)))

            f.create_dataset(
                str(img_id), data=features, compression="gzip", chunks=True
            )
            f.attrs["num_boxes"] = num_boxes
    
    f.close()

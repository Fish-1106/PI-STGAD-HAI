# HAI data placement

Download HAI 21.03 from the dataset maintainers’
[official repository](https://github.com/icsdataset/hai/tree/master/hai-21.03).

Place the files used by the public workflow under:

```text
data/
└── hai-21.03/
    ├── train1.csv
    ├── train2.csv
    ├── train3.csv
    └── test5.csv
```

Run the preparation command from the repository root:

```bash
python scripts/prepare_hai_inputs.py --hai-root data/hai-21.03
```

The `data/hai-21.03/` directory is excluded from Git.

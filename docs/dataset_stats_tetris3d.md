# Dataset audit: tetris3d

Generator version: `2`
Dataset hash: `9a8d20c45d58272535863ab9ea5418c85a9a4ef9572483aa9483ef73d3e6e94f`
Config hash: `0ffe45498b9f8a93d1bc4a63dd429bffc546b5d4d8d6a05097a9d4ea8ba29d55`
Pillow version: `12.3.0`

Hashes cover UTF-8 symbolic JSONL bytes. Rendered pixels are not hashed.

| split | pool | samples | cell counts | steps | answer counts | hash ok |
|---|---|---|---|---|---|---|
| train | train | 16000 | 4:5319, 5:5268, 6:5413 | 1:8080, 2:7920 | 4076/4019/3977/3928 | yes |
| sc_id_test | train | 400 | 4:138, 5:130, 6:132 | 1:206, 2:194 | 105/100/100/95 | yes |
| c_id_test | heldout | 400 | 4:124, 5:129, 6:147 | 1:195, 2:205 | 83/116/95/106 | yes |
| ood_test | ood | 500 | 7:500 | 1:261, 2:239 | 111/130/123/136 | yes |
| depth3_test | train | 400 | 4:130, 5:133, 6:137 | 3:400 | 111/95/88/106 | yes |

Duplicate symbolic questions: 0
Reference and query membership and mirror-separated shape pools checked.
Regeneration audit: 17700/17700 samples reproduced exactly

## Result

PASS

# Dataset audit: tetris2d

Generator version: `2`
Dataset hash: `2af1bbe495d685c450bfd40a7edf658d507bad164678bcce1bc80c6049962bec`
Config hash: `16b1c4bef85598a1037735d4c7f8e5a8baafa9c3b2b89a61cc2921a439be9de1`
Pillow version: `12.3.0`

Hashes cover UTF-8 symbolic JSONL bytes. Rendered pixels are not hashed.

| split | pool | samples | cell counts | steps | answer counts | hash ok |
|---|---|---|---|---|---|---|
| train | train | 4000 | 4:1263, 5:1343, 6:1394 | 1:2419, 2:1581 | 991/1014/990/1005 | yes |
| id_test | train | 400 | 4:126, 5:126, 6:148 | 1:222, 2:178 | 98/94/97/111 | yes |
| ood_test | ood | 500 | 7:500 | 1:304, 2:196 | 110/125/132/133 | yes |
| depth3_test | train | 400 | 4:133, 5:136, 6:131 | 3:400 | 108/104/99/89 | yes |

Duplicate symbolic questions: 0
Reference and query membership and mirror-separated shape pools checked.
Regeneration audit: 5300/5300 samples reproduced exactly

## Result

PASS

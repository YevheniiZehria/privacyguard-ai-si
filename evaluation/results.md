# PrivacyGuard evaluation report

Dataset: 12 synthetic documents (4 fictional people). Total runtime 29s.

## PII detection
| Type            |   GT |   Detected |   Precision |   Recall |   F1 |
|-----------------|------|------------|-------------|----------|------|
| CNP             |    8 |          8 |           1 |        1 |    1 |
| PERSON_NAME     |   24 |         24 |           1 |        1 |    1 |
| ADDRESS         |   20 |         12 |           1 |        1 |    1 |
| ID_SERIES       |    8 |         11 |           1 |        1 |    1 |
| IBAN            |    4 |          4 |           1 |        1 |    1 |
| PHONE           |    8 |          8 |           1 |        1 |    1 |
| EMAIL           |    8 |          8 |           1 |        1 |    1 |
| BIRTH_DATE      |    4 |          4 |           1 |        1 |    1 |
| MRZ             |    8 |          8 |           1 |        1 |    1 |
| **ALL (micro)** |   92 |         87 |           1 |        1 |    1 |

Also redacted (not in ground truth, not scored): ID_DATE 4, ISSUER 4

## Redaction
* Ground-truth PII values still readable after redaction: **0 / 92** (redaction recall 1.000)
* verify_redaction PASS: **12 / 12**

| File              | Kind    | Font    |   Rot (deg) |   OCR conf |   GT PII |   Detected |   Boxes |   GT leaks | Verify   | Time   |
|-------------------|---------|---------|-------------|------------|----------|------------|---------|------------|----------|--------|
| 01_id_card_P1.png | id_card | arial   |         0   |      92.49 |        8 |         10 |      10 |          0 | PASS     | 2.3s   |
| 02_form_P1.png    | form    | times   |         1.5 |      93.81 |        9 |          8 |       9 |          0 | PASS     | 2.5s   |
| 03_invoice_P1.jpg | invoice | courier |        -2   |      92.72 |        6 |          6 |       6 |          0 | PASS     | 2.3s   |
| 04_id_card_P2.png | id_card | calibri |         3   |      89.28 |        8 |          9 |      10 |          0 | PASS     | 2.1s   |
| 05_form_P2.png    | form    | verdana |        -1   |      93.87 |        9 |          8 |       9 |          0 | PASS     | 2.1s   |
| 06_invoice_P2.jpg | invoice | georgia |         2.5 |      93.11 |        6 |          6 |       6 |          0 | PASS     | 2.3s   |
| 07_id_card_P3.png | id_card | tahoma  |        -3.5 |      91.55 |        8 |         10 |      10 |          0 | PASS     | 2.9s   |
| 08_form_P3.png    | form    | arial   |         0.8 |      94.17 |        9 |          8 |       9 |          0 | PASS     | 2.5s   |
| 09_invoice_P3.jpg | invoice | times   |        -2.8 |      91.92 |        6 |          6 |       6 |          0 | PASS     | 2.3s   |
| 10_id_card_P4.png | id_card | courier |         1.2 |      92.47 |        8 |         10 |      10 |          0 | PASS     | 2.8s   |
| 11_form_P4.png    | form    | calibri |         4   |      93.96 |        9 |          8 |       9 |          0 | PASS     | 2.8s   |
| 12_invoice_P4.jpg | invoice | verdana |        -0.6 |      92.85 |        6 |          6 |       6 |          0 | PASS     | 2.3s   |

## Cross-document linking
|   GT pairs |   Pred pairs |   Precision |   Recall |   F1 |   Clusters |
|------------|--------------|-------------|----------|------|------------|
|         12 |           12 |           1 |        1 |    1 |          4 |

* cluster_1: 3 docs, persons ['P3'], types ['CNP', 'ID_SERIES', 'PERSON_NAME', 'PHONE'], strength 0.6
* cluster_2: 3 docs, persons ['P1'], types ['CNP', 'ID_SERIES', 'PERSON_NAME', 'PHONE'], strength 0.6
* cluster_3: 3 docs, persons ['P2'], types ['CNP', 'ID_SERIES', 'PERSON_NAME', 'PHONE'], strength 0.6
* cluster_4: 3 docs, persons ['P4'], types ['CNP', 'EMAIL', 'ID_SERIES', 'PERSON_NAME', 'PHONE'], strength 0.6

### Details: https://spacy.io/models/mk#mk_core_news_sm

Macedonian pipeline optimized for CPU. Components: tok2vec, morphologizer, parser, senter, ner, attribute_ruler, lemmatizer.

| Feature | Description |
| --- | --- |
| **Name** | `mk_core_news_sm` |
| **Version** | `3.8.0` |
| **spaCy** | `>=3.8.0,<3.9.0` |
| **Default Pipeline** | `morphologizer`, `parser`, `attribute_ruler`, `lemmatizer`, `ner` |
| **Components** | `morphologizer`, `parser`, `senter`, `attribute_ruler`, `lemmatizer`, `ner` |
| **Vectors** | 0 keys, 0 unique vectors (0 dimensions) |
| **Sources** | [Macedonian Corpus](https://blog.netcetera.com/macedonian-spacy-f3c85484777f) (Damjan Zlatinov, Melanija Gerasimovska, Borijan Georgievski, Marija Todosovska)<br />[spaCy lookups data](https://github.com/explosion/spacy-lookups-data) (Explosion) |
| **License** | `CC BY-SA 4.0` |
| **Author** | [Explosion](https://explosion.ai) |

### Label Scheme

<details>

<summary>View label scheme (54 labels for 3 components)</summary>

| Component | Labels |
| --- | --- |
| **`morphologizer`** | `POS=PROPN`, `POS=AUX`, `POS=ADJ`, `POS=NOUN`, `POS=ADP`, `POS=PUNCT`, `POS=CONJ`, `POS=NUM`, `POS=VERB`, `POS=PRON`, `POS=ADV`, `POS=SCONJ`, `POS=PART`, `POS=SYM`, `_`, `POS=SPACE`, `POS=X`, `POS=INTJ` |
| **`parser`** | `ROOT`, `advmod`, `att`, `aux`, `cc`, `dep`, `det`, `dobj`, `iobj`, `neg`, `nsubj`, `pobj`, `poss`, `pozm`, `pozv`, `prep`, `punct`, `relcl` |
| **`ner`** | `CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART` |

</details>

### Accuracy

| Type | Score |
| --- | --- |
| `TOKEN_ACC` | 100.00 |
| `TOKEN_P` | 100.00 |
| `TOKEN_R` | 100.00 |
| `TOKEN_F` | 100.00 |
| `SENTS_P` | 56.96 |
| `SENTS_R` | 58.44 |
| `SENTS_F` | 57.69 |
| `ENTS_P` | 72.29 |
| `ENTS_R` | 71.06 |
| `ENTS_F` | 71.67 |
| `POS_ACC` | 91.93 |
| `DEP_UAS` | 65.88 |
| `DEP_LAS` | 47.70 |
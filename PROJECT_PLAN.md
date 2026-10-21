# ArXiv Curator — Requisiti e Piano di Azione

> Documento di contesto del progetto. Scopo: permettere a chiunque (umano o agente AI)
> di riprendere il lavoro da zero leggendo solo questo file. Aggiornarlo a ogni decisione
> architetturale rilevante e a ogni fase completata.

Ultimo aggiornamento: 2026-10-21 — Stato: **Fase 1 e Fase 2 (Ingestion + Storage Delta/Volume) completate**.

---

## 1. Obiettivo del progetto

Sistema **agentico** (pipeline deterministica con **tool calling**) su Databricks che:

1. Riceve in input un **argomento** (query di ricerca) e dei **parametri**:
   - `num_papers`: numero N di paper da selezionare (top-N)
   - `ranking_criteria`: criterio di selezione — di default `relevance`, modificabile in
     `lastUpdatedDate` o `submittedDate` (con parametro `date_filter` opzionale quando si filtra per data).
     *(Estensione futura: modulo separato per calcolare l'importanza tramite citazioni con tool esterni).*
2. Scarica da **arXiv** i top-N paper secondo il criterio (metadati + PDF).
3. Arricchisce i paper con un **LLM** via AI Gateway del corso (usando **tool calling / structured outputs** per estrarre tag tematici normalizzati, riassunto breve ed eventuali query/metadati).
4. Costruisce un **knowledge graph** che collega le entità tramite:
   - `SHARES_TAG`: collega paper che condividono tag/categorie
   - `CITES`: citazioni dirette tra paper (quando disponibili dai metadati/riferimenti)
   - `AUTHORS`: collega un nodo Autore (`Author`) al paper scritto (`Paper`)
   - `SHARED_AUTHORS`: collega paper che hanno uno o più co-autori in comune
5. Salva i dati strutturati in **Delta Tables** su Unity Catalog e gli **embeddings** in **Lakebase (`pgvector`)**.
6. Esporta il risultato nel **Volume Unity Catalog** sia come **cartella pronta per Obsidian** (vault) sia come **archivio `.zip` scaricabile**.

### Estensibilità richiesta (design driver)

Il sistema deve essere pronto, senza refactoring strutturali, a supportare:

- **Altri indici bibliografici**: PubMed come primo esempio → pattern *provider* (`PaperSource`).
- **Calcolo importanza da citazioni**: modulo/tool separato aggiungibile in futuro per ri-ordinare o filtrare i paper in base all'impatto citazionale.
- **Ranking dei journal**: modulo di scoring separato che consuma i metadati raccolti.
- Nuovi criteri di selezione e nuovi tipi di nodi/edge nel grafo.

---

## 2. Vincoli: cosa NON toccare

- **Non modificare le versioni dell'ambiente**: `pyproject.toml` ha dipendenze pinnate e
  `requires-python = "==3.12.*"`. Aggiungere una dipendenza solo se strettamente necessaria
  e discuterla prima con l'utente; rigenerare `uv.lock` solo in quel caso.
- **Mantenere la struttura Databricks Asset Bundle**: `databricks.yml`, `resources/`,
  i tre target `dev`/`acc`/`prd` e il job `hello` esistente restano come sono.
  Le nuove risorse si **aggiungono**, non sostituiscono.
- Le regole ruff (`pyproject.toml`) e la CI (`.github/workflows/ci.yml`, pre-commit) restano
  attive: ogni nuovo codice deve passare lint + test.
- `project_config.yml` è la **fonte unica di verità** per catalog/schema/endpoint: mai
  hardcodare nomi di risorse nel codice, usare sempre `arxiv_curator.config.load_config`.
- `src/arxiv_curator/config.py` esiste già: non cambiarne l'API pubblica.
- Nota nota: `environment_version` è `"5"` in `pyproject.toml` e `"6"` in
  `resources/hello.job.yml`. Non "correggere": verificare col corso prima di toccarli.

### Risorse provisionate dal corso (da `project_config.yml` / README)

| Risorsa | Valore |
| --- | --- |
| Schemi UC | `llmops_{dev,acc,prd}.riccardo_tambone` (dev scrivibile; acc/prd read-only per me) |
| Volume suggerito | `arxiv_files` (da creare, es. `llmops_dev.riccardo_tambone.arxiv_files`) |
| LLM chat (AI Gateway) | `course_ops.gateway.chat` → GPT-OSS-120B, budget mensile per studente |
| Embeddings (AI Gateway) | `course_ops.gateway.embed` → GTE Large |
| SQL warehouse | `ce85f15d22846b60` |
| MLflow experiments | `/llmops/riccardo_tambone-{dev,acc,prd}` |
| Lakebase | `lb-llmops-riccardo-tambone` (condiviso tra env, 0.5–2 CU, suspend dopo 5 min idle; database e pgvector **da creare**) |
| Usage/budget policy id | `891ab3b2-3b04-3ccb-8359-9327a16ef424` |

Chiamate ai modelli: pacchetto `openai` (già nelle deps) con base URL del workspace
`/ai-gateway/mlflow/v1` e auth Databricks. `databricks-openai` NON è nelle deps — valutare
prima di aggiungerlo.

### Dipendenze già disponibili (rilevanti per il progetto)

`arxiv==2.3.1`, `pydantic`, `loguru`, `backoff`, `mlflow`, `openai`, `databricks-sdk`,
`databricks-agents`, `psycopg[binary]` + `psycopg-pool` (per Lakebase/pgvector),
`databricks-vectorsearch`, `pandas`, `pyarrow`.

---

## 3. Criteri di selezione e gestione metadati

- **Criteri nativi supportati subito (arXiv API)**:
  - `relevance` (**default**): ordinamento per rilevanza di arXiv (`arxiv.SortCriterion.Relevance`).
  - `lastUpdatedDate`: ordinamento per data di ultimo aggiornamento (`arxiv.SortCriterion.LastUpdatedDate`), con possibilità di passare un parametro data (`date_from` / `date_to` o data di riferimento).
  - `submittedDate`: ordinamento per data di prima sottomissione (`arxiv.SortCriterion.SubmittedDate`), con possibilità di passare un parametro data.
- **Modulo futuro per citazioni / importanza**:
  - Predisporre l'interfaccia `CitationEnricher` / `ImportanceScorer` (es. via Semantic Scholar / OpenAlex) come tool opzionale per arricchire gli archi `CITES` e calcolare uno score di importanza in una fase successiva.
- **Mai inventare citazioni via LLM**: gli edge `CITES` devono derivare da riferimenti verificabili (es. estrazione arXiv ID / DOI o provider citazionale).

---

## 4. Architettura proposta (Pipeline deterministica + Tool Calling)

```text
┌─────────────────────────────────────────────────────────────────────┐
│                  JOB BUNDLE (parametrizzato, nuovo)                 │
│  query, num_papers, ranking_criteria (default: relevance),          │
│  target_date (opz.), env                                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. INGESTION — sources/ (pattern Provider + Tool)                  │
│     ArxivSource.search(query, criteria, top_n, target_date)         │
│     + download PDF → UC Volume arxiv_files                          │
│     + salvataggio metadati grezzi/normalizzati → Delta table        │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. AGENT / ENRICHMENT (Pipeline deterministica con Tool Calling)   │
│     - LLM (course_ops.gateway.chat) invoca tool strutturati per:    │
│       1) raffinare/costruire query e filtri (opz.)                  │
│       2) estrarre tag tematici normalizzati + riassunto per paper   │
│     - Embedding abstract (course_ops.gateway.embed → GTE Large)     │
│     Metadati/Tag → Delta tables  |  Vettori → Lakebase pgvector     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. GRAPH BUILDER — graph/                                          │
│     Nodi: Paper, Author, Tag                                        │
│     Edge:                                                           │
│       - SHARES_TAG (paper ↔ paper o paper → tag)                    │
│       - AUTHORS (author → paper)                                    │
│       - SHARED_AUTHORS (paper ↔ paper con co-autori comuni)         │
│       - CITES (paper → paper citato)                                │
│     Grafo materializzato in Delta tables (graph_nodes, graph_edges) │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. OBSIDIAN EXPORT — export/                                       │
│     - Note .md per Paper (frontmatter YAML con #tags + [[wikilinks]])│
│     - Note .md per Tag (tags/<tag>.md) e Autori/MOC (index.md)      │
│     - Nessun graph.canvas (si usa il Graph View nativo di Obsidian) │
│     - Scrittura su UC Volume: cartella vault + archivio .zip        │
└─────────────────────────────────────────────────────────────────────┘
```

### Struttura del package `src/autographrs/`

> Nota: `src/arxiv_curator/config.py` rimane intatto per il job di test `hello` e i test base del
> corso, mentre tutto il codice del nostro sistema vive in `src/autographrs/` (incluso nel wheel in
> `pyproject.toml`).

```text
autographrs/
├── config.py                # Re-export/wrapper di ProjectConfig, load_config, get_env
├── models.py                # Pydantic: Paper, Author, Tag, GraphNode, GraphEdge, SearchRequest
├── sources/
│   ├── base.py              # Protocol/ABC PaperSource (punto di estensione per PubMed, ecc.)
│   └── arxiv.py             # ArxivSource (relevance, lastUpdatedDate, submittedDate + filtro data)
├── enrichment/
│   ├── llm.py               # Client AI Gateway (chat + embeddings) + definizione Tools
│   ├── tagger.py            # Step deterministico con tool-calling per estrazione tag e summary
│   └── prompts/             # System/user prompt versionati
├── storage/
│   ├── delta.py             # Lettura/scrittura Delta tables su Unity Catalog
│   └── vector_db.py         # Gestione Lakebase Postgres + pgvector per embeddings
├── graph/
│   └── builder.py           # Costruzione nodi e archi: SHARES_TAG, CITES, AUTHORS, SHARED_AUTHORS
├── export/
│   └── obsidian.py          # Generazione vault Markdown (#tags + note tags/ + wikilinks) + .zip
├── ranking/
│   ├── importance.py        # (ESTENSIONE FUTURA) calcolo importanza da citazioni
│   └── journals.py          # (ESTENSIONE FUTURA) scoring venue/journal
└── pipeline.py              # Orchestrazione deterministica dei passi 1→4, entrypoint del job
```

Decisione chiave: **`PaperSource` è il punto di estensione** — PubMed sarà
`PubMedSource(PaperSource)` senza toccare la pipeline. Allo stesso modo, `ranking/importance.py`
e `ranking/journals.py` potranno essere agganciati come step/tool aggiuntivi.

---

## 5. Data model e storage

| Cosa | Dove | Note |
| --- | --- | --- |
| PDF dei paper | UC Volume `arxiv_files` | `<volume>/runs/<run_id>/pdfs/<arxiv_id>.pdf` |
| Metadati paper | Delta table `papers` | `arxiv_id`, `title`, `authors`, `abstract`, `published`, `updated`, `categories`, `pdf_path`, `run_id` |
| Tag/riassunti LLM | Delta table `paper_enrichment` | `arxiv_id`, `tags` (array), `summary`, `run_id` |
| Nodi e Archi grafo | Delta tables `graph_nodes`, `graph_edges` | Archi tipizzati: `SHARES_TAG`, `CITES`, `AUTHORS`, `SHARED_AUTHORS` |
| Embeddings | Lakebase (`pgvector`) | Tabella `paper_embeddings` su `lb-llmops-riccardo-tambone` |
| Vault Obsidian + ZIP | UC Volume `arxiv_files` | `<volume>/runs/<run_id>/obsidian_vault/` + `<volume>/runs/<run_id>/obsidian_vault.zip` |

Ogni run ha un `run_id` (timestamp + slug argomento) → run ripetibili, storico preservato nelle
Delta tables e nel Volume.

### Vault Obsidian (output finale)

```text
obsidian_vault/
├── index.md                 # MOC (Map of Content): riepilogo run, lista paper, autori e tag
├── papers/
│   └── 2301.12345.md        # Nota paper: frontmatter YAML (#tags nativi), abstract, summary,
│                            # link [[tags/...]], [[authors/...]], CITES e SHARED_AUTHORS
├── tags/
│   └── rag.md               # Nota dedicata al tag → elenca e linka i paper con quel tag
└── authors/
    └── Nome Cognome.md      # (opz./leggera) Nota autore o sezione per navigare archi AUTHORS
```

Modalità doppia per i Tag (come deciso):

1. **`#tag` nativi Obsidian** nel frontmatter YAML (`tags: [llm, rag, ...]`) per filtraggio nativo.
2. **Note dedicate in `tags/<tag>.md`** linkate con `[[tags/rag|rag]]` per vedere i nodi Tag anche
   esplicitamente e aggiungere descrizioni/appunti sul concetto.

---

## 6. Asset Bundle: nuove risorse (senza toccare l'esistente)

Nuovo file `resources/paper_curator.job.yml`:

- Job `job-ingest-papers`, serverless,
  `environment_version: "5"` (coerente con `pyproject.toml`), dipendenza dal wheel
  `../dist/*.whl`, `budget_policy_id: "891ab3b2-3b04-3ccb-8359-9327a16ef424"`.
- Task notebook `resources/deployment_notebooks/paper_curator.py` con parametri widget:
  - `query`: argomento di ricerca
  - `num_papers`: numero di paper (default `10`)
  - `ranking_criteria`: `relevance` (default) | `lastUpdatedDate` | `submittedDate`
  - `target_date`: data opzionale (es. `YYYY-MM-DD` o range) quando si usa `lastUpdatedDate` / `submittedDate`
  - `env`: `${bundle.target}`
- Il job di test `hello` (`resources/hello.job.yml`) resta intatto.

---

## 7. Roadmap per fasi

| Fase | Contenuto | Criterio di completamento | Stato |
| --- | --- | --- | --- |
| **0 — Setup infrastruttura** | Creare volume UC `arxiv_files`; configurare database su Lakebase e abilitare estensione `pgvector` | Volume e tabella `pgvector` raggiungibili da dev | Volume gestito via `ensure_volume_exists()`; Lakebase da fare in Fase 3 |
| **1 — Dominio & Ingestion arXiv** | `models.py`, `sources/base.py`, `sources/arxiv.py` (`relevance`, `lastUpdatedDate`, `submittedDate` + filtro data); test unitari offline | `ArxivSource` ritorna top-N paper + scarica PDF; pytest e ruff verdi | ✅ Completata |
| **2 — Persistenza Delta & Volume** | `storage/delta.py` (tabella `papers` + Volume UC), `pipeline.py` (`run_ingestion`), job `ingest_papers.job.yml` | Scrittura/lettura Delta e salvataggio PDF su Volume funzionanti e testati | ✅ Completata |
| **3 — Enrichment con Tool Calling** | `enrichment/llm.py` + `tagger.py`: chiamata a `course_ops.gateway.chat` con **tool calling** per estrarre tag/summary strutturati + embeddings con `course_ops.gateway.embed` su Lakebase + MLflow tracing | Tag, summary ed embeddings generati e salvati; tracciati su MLflow | ⏳ Prossima |
| **4 — Knowledge Graph Builder** | `graph/builder.py`: generazione archi `SHARES_TAG`, `AUTHORS`, `SHARED_AUTHORS`, `CITES` e salvataggio su Delta tables | Tabelle `graph_nodes` e `graph_edges` popolate correttamente | ⏳ Da fare |
| **5 — Esportatore Obsidian + ZIP** | `export/obsidian.py`: generazione note paper, note `tags/`, `authors/`, `index.md` (con `#tags` + `[[wikilinks]]`, senza `.canvas`) e creazione `.zip` nel Volume UC | Cartella e file `.zip` creati nel Volume UC e navigabili in Obsidian | ⏳ Da fare |
| **6 — Pipeline End-to-End & Bundle** | Estensione `pipeline.py` a tutti gli step 1→4 | `databricks bundle validate -t dev` e run end-to-end su dev completati | ⏳ Da fare |
| **7 — Estensioni future** | Modulo calcolo importanza da citazioni (`ranking/importance.py`), nuovo indice `PubMedSource`, ranking dei journal (`ranking/journals.py`) | Estensioni aggiunte tramite i contratti base senza modificare il core | ⏳ Da fare |

---

## 8. Aspetti LLMOps (richiesti dal corso)

- **MLflow Tracking & Tracing**: loggare per ogni run su `/llmops/riccardo_tambone-{env}`:
  parametri (`query`, `num_papers`, `ranking_criteria`, `target_date`), metriche (n. paper
  scaricati, n. nodi/archi per tipo, token consumati), tracce delle chiamate tool-calling LLM e
  path del `.zip` generato.
- **Tool Calling deterministico**: l'LLM non esegue codice libero; la pipeline gli fornisce
  schema/tool definiti (es. `record_paper_analysis`, `extract_citations`) validati con Pydantic.
- **Prompt Versioning**: prompt salvati in `enrichment/prompts/` e tracciati con la run.
- **Resilienza**: `backoff` sulle chiamate API arXiv e AI Gateway; se l'LLM fallisce su un singolo
  paper, il paper viene comunque incluso nel grafo usando le categorie arXiv native.

---

## 9. Decisioni architetturali consolidate

1. **Architettura agentica**: **Pipeline deterministica con Tool Calling** (niente loop autonomi imprevedibili; step chiari e tracciabili su MLflow).
2. **Ordinamento e selezione**: default `relevance`, selezionabili `lastUpdatedDate` e `submittedDate` (con parametro data). Il calcolo dell'importanza tramite citazioni sarà un modulo/tool futuro separato.
3. **Storage**: **Opzione A** — Metadati, arricchimenti e grafo salvati in **Delta Tables** su Unity Catalog; embeddings salvati su **Lakebase (`pgvector`)**.
4. **Relazioni del Knowledge Graph**: `SHARES_TAG`, `CITES`, `SHARED_AUTHORS`, `AUTHORS`.
5. **Formato Obsidian**: sia `#tag` YAML nativi sia note dedicate per tag (`[[tags/...]]`), **nessun** file `graph.canvas`, salvataggio su Volume UC con generazione di archivio **`.zip` scaricabile**.
6. **Ambiente Databricks**: `hello.job.yml` lasciato intatto come job di test; il nuovo job userà la stessa configurazione serverless.

---

## 10. Istruzioni per riprendere il contesto

1. Rileggere questo file per intero.
2. Controllare `git status` per modifiche dell'utente: non sovrascriverle.
3. Verificare lo stato delle fasi nella tabella §7 (aggiornarla a ogni avanzamento).
4. Non considerare implementato nulla di pianificato finché non è nel codice con test.
5. Prima di cambiare versioni, lockfile, target del bundle o risorse provisionate:
   chiedere all'utente.

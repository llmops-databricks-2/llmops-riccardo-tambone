# llmops-riccardo-tambone

Your repository for the LLMOps course on Databricks. Everything you need in
the course workspace was provisioned for you before you got this repo, and every id is already in
`project_config.yml`: you configure nothing.

## What was provisioned for you

| What | Value |
|---|---|
| Your schema in development, which you manage | `llmops_dev.riccardo_tambone` |
| Your schema in acceptance, which you read | `llmops_acc.riccardo_tambone` |
| Your schema in production, which you read | `llmops_prd.riccardo_tambone` |
| Suggested volume name (create during the course) | `arxiv_files`, such as `llmops_dev.riccardo_tambone.arxiv_files` |
| SQL warehouse id | `ce85f15d22846b60` |
| Usage policy id | `891ab3b2-3b04-3ccb-8359-9327a16ef424` |
| MLflow experiments | Dev: `/llmops/riccardo_tambone-dev`; acc: `/llmops/riccardo_tambone-acc`; prd: `/llmops/riccardo_tambone-prd` |
| Chat model service | `course_ops.gateway.chat` |
| Embedding model service | `course_ops.gateway.embed` |

Both settings name model services of the course’s AI Gateway, with a monthly budget per student.
Chat routes to GPT-OSS-120B; embeddings route to GTE Large. Call both through the gateway
with the `databricks-openai` package:

```python
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI(use_ai_gateway=True)
client.chat.completions.create(
    model="course_ops.gateway.chat",
    messages=[...],
    reasoning_effort="low",
    max_tokens=1024,
)
client.embeddings.create(model="course_ops.gateway.embed", input="A paper abstract")
```

The `databricks-openai` package is not included in this repo’s dependencies. The same APIs
are available through the installed `openai` client with the workspace base URL
`/ai-gateway/mlflow/v1` and Databricks authentication.

`embedding_endpoint` is a gateway model service despite the existing field name. Generate
vectors through its embeddings API before storing them in pgvector. It is not a managed
Vector Search index’s `embedding_model_endpoint_name`.

## Lakebase

The `lakebase_instance` value in `project_config.yml` identifies your Lakebase project.
All three environments share it; your Unity Catalog schemas remain separate.
Onboarding does not create an application database or install pgvector. Create the database
and extensions you need during the course. A Vector Search endpoint is not provisioned.

For a dedicated student project, you and your three deployment service principals have project
management access and OAuth Postgres roles with database administration privileges. The project
carries your usage policy and student cost tag. Its initial compute range is 0.5–2 CU, with
suspension after five idle minutes. Creating extra branches or computes can increase its cost.

## What is in the repo

- `src/arxiv_curator/`: starts with `config.py`, which loads `project_config.yml`.
  Add the other course modules as you work through the lectures.
- `lecture_1/`, `lecture_2/` and so on, when you add them: copy a lecture's folder from the course
  code as it is, to the root of this repo, where its notebooks find `project_config.yml` one level up.
  Commit it with the hooks installed (below): the first commit stops once, after the hooks have
  formatted the notebooks and fixed what ruff can, and goes through when you add their changes and
  commit again. Ruff rules are configured in `pyproject.toml`.
- `databricks.yml` and `resources/`: the bundle, with three targets, `dev` (your sandbox, in
  development mode, the default), `acc` and `prd`. It builds the package as a wheel with `uv build`
  and deploys the `hello` job, named after its target (`job-hello-riccardo-tambone-dev`,
  `-acc`, `-prd`), one serverless notebook task that installs the wheel and writes one row to the
  `hello` table of your schema in the target's catalog, such as
  `llmops_dev.riccardo_tambone.hello`. The course's onboarding
  does not deploy these jobs yet. Deployment workflows are introduced in a later lecture.
- `pyproject.toml` and `version.txt`: the package and its version, bumped as the lectures ask.
- `tests/`: your tests; `test_config.py` checks that `project_config.yml` loads.
- `.github/workflows/ci.yml`: on every pull request and push to `main`, installs the package with
  uv and runs the pre-commit hooks and the tests.
- `.github/CODEOWNERS`: who reviews a pull request to `main`, which needs one approving review.

## Getting started

```bash
uv sync --extra dev
uvx pre-commit install
databricks bundle validate -t dev
```

Each environment logs to its own MLflow experiment and UC trace tables in its matching catalog. You can manage the dev experiment and write dev traces; acc and prd are read-only for you. Each deploy service principal manages its own environment’s experiment and writes to that environment’s schema.

## AutoGraphRS citation enrichment

Il metodo predefinito usa **Semantic Scholar in batch**, con **OpenAlex come fallback**
solo per i paper per cui Scholar non ha restituito alcuna voce bibliografica.
Un risultato Scholar parziale ma con riferimenti viene conservato, non sostituito.
Non e' una fusione delle bibliografie dei due indici e non richiede HTML arXiv.

Scholar riceve blocchi da 100 identificativi per richiesta (`batch_size`, massimo API 500):
il limite ufficiale di 10 MB per risposta verrebbe superato con 500 paper e riferimenti inline.
Richiede riferimenti con titolo,
autori, anno, DOI, ID Scholar e ID arXiv quando presenti. Le liste troncate vengono paginate.
`paper_enrichment.bibliography` conserva anche citati fuori da arXiv e riferimenti senza
identificatori, questi ultimi come placeholder locali al paper citante. Non scarica
PDF esterni, non espande ricorsivamente le bibliografie e non genera ancora note.
`text` e' il titolo restituito dall'indice, o l'ID se mancante: non la citazione testuale
originale del PDF. `evidence_url` punta al lavoro citante nell'indice utilizzato.

### Retry e autenticazione Scholar

- Richieste seriali distanziate di almeno 1,1 secondi; fino a 4 tentativi per richiesta.
- Su 429 senza `Retry-After`, attese di circa 10, 20 e 40 secondi piu' jitter.
  Su errori di trasporto/5xx, backoff da 2 secondi. Nessun retry dei normali 4xx permanenti.
- `Retry-After` numerico o HTTP-date viene rispettato; se supera 60 secondi o il budget
  residuo, si interrompe invece di richiamare prematuramente.
- Budget condiviso di **90 secondi di sole attese retry per chiamata bulk**: non viene
  azzerato a ogni pagina. Timeout e pacing sono aggiuntivi, quindi non e' un limite
  di 90 secondi all'intera fase.
- Dopo un batch che esaurisce i tentativi, i batch restanti vengono marcati senza ripetere
  il ciclo per ogni paper. Dopo una paginazione fallita si conservano i riferimenti gia'
  acquisiti e si evitano nuove paginazioni nella stessa chiamata bulk.
- Configurare preferibilmente `SEMANTIC_SCHOLAR_API_KEY` nel processo del job tramite
  il meccanismo secret approvato. Mai in Git, nei parametri job o nei log. La quota anonima
  e' condivisa: il retry aiuta ma non garantisce disponibilita'. Il token Databricks
  non autentica Scholar. Vedi [API Scholar](https://www.semanticscholar.org/product/api).

OpenAlex risolve tramite location arXiv HTTP/HTTPS/DOI arXiv o DOI disponibile; per altre
fonti usa DOI o landing URL esatto. Recupera i metadati dei citati in batch fino a 100 ID,
con cache. `OPENALEX_API_KEY` e' opzionale, sempre da secret del processo.
Le quote restano quelle di [OpenAlex](https://help.openalex.org/api/authentication/).

`citation_source` indica il provider del risultato scelto: `semantic_scholar` o `openalex`.
`citation_status` distingue `ok`, `partial`, `not_found`, `rate_limited`, `error`, `disabled`.
`citation_lookups` conserva **separatamente l'esito di ogni provider chiamato**, con fonte,
stato, errore e timestamp, anche quando il fallback riesce. Gli esiti non riusciti producono
warning nella run; non vengono nascosti da un risultato positivo del fallback.
Riferimenti senza identita', paginazione incompleta, metadati OpenAlex mancanti o zero
riferimenti indicizzati producono `partial`. Un paper non trovato e' `not_found`.
Una lista vuota non prova l'assenza di bibliografia; `ok` non certifica la copertura
completa dell'indice o la disponibilita' di ogni metadato.

L'enrichment si ferma a **`bibliography`** e non costruisce relazioni tra paper.
Il futuro graph builder leggera' la bibliografia per costruire archi, compresi quelli
verso opere esterne alla selezione. Non e' ancora implementato.
Le vecchie colonne `referenced_arxiv_ids`, `cites_in_set`, `cited_by_in_set` e
`papers.cited_paper_ids` non vengono piu' scritte o lette dall'applicazione.
Tabelle Delta gia' esistenti possono mantenerle: i valori storici, inclusi quelli di
run ritentate, non vengono cancellati o aggiornati e non vanno usati come grafo attuale.
Le nuove tabelle non le contengono; nuove righe in tabelle esistenti le avranno null.
Lo schema Spark esplicito e l'evoluzione Delta includono i nuovi metadati nested;
le vecchie righe restano leggibili con i nuovi attributi null, senza backfill automatico.
Avviare una **nuova run** per provare il metodo; una run completata non viene riarricchita.

I resolver Scholar, OpenAlex e HTML arXiv restano selezionabili esplicitamente in Python.
Il job predefinito non usa HTML come fallback.
Il metodo HTML conserva le voci `ltx_bibitem` e il testo originale normalizzato, anche senza
identificatori; un 404 segnala HTML assente, limite che non riguarda il nuovo metodo OpenAlex.
Gli exporter sono ancora da implementare, come descritto in [PROJECT_PLAN.md](PROJECT_PLAN.md).

### Compatibilita' con altre fonti

La fonte di ricerca (`PaperSource`, per esempio arXiv o un futuro provider PubMed) e il
provider di citazioni (Scholar/OpenAlex) hanno responsabilita' separate. Il percorso principale
passa al resolver l'intero `Paper`: `source`, `paper_id`, DOI e URL della pagina.

- Scholar usa `ARXIV:<id>` per arXiv; per altre fonti DOI, altrimenti PMID per PubMed
  quando numerico, altrimenti URL del paper. Tutti gli identificativi vanno nello stesso batch.
- OpenAlex mantiene i match specifici arXiv; per altre fonti usa DOI o URL esatto,
  verificando la corrispondenza nella risposta.
- Nessun ID locale di un altro repository viene interpretato come ID arXiv; nessun match
  basato soltanto sul titolo. URL non indicizzati o DOI assenti possono impedire la risoluzione.
- I riferimenti mantengono target Scholar/OpenAlex/DOI indipendenti dalla fonte iniziale.
  La bibliografia e' il dato di ingresso del futuro grafo.
- Le categorie di altre fonti rimangono in `papers.categories`: non vengono convertite con
  la tassonomia arXiv. In quel caso `arxiv_category` e' vuoto e i tag LLM restano disponibili.

Il job cerca ancora su arXiv: il provider di ricerca PubMed non e' implementato.
La compatibilita' generica e' verificata con test di paper non-arXiv e persistenza end-to-end.
Ogni run usa una sola fonte di ricerca; non e' ancora una ricerca federata multi-repository.
Il provider HTML, arXiv-only, viene rifiutato esplicitamente su paper di altre fonti.

## Dati

Ogni ricerca produce un **dataset autonomo identificato da `run_id`**. AutoGraphRS salva
tre tabelle Delta nello schema Unity Catalog dell'ambiente selezionato e i PDF nel Volume
`arxiv_files`. In dev le tabelle sono in `llmops_dev.riccardo_tambone`; acc e prd usano
i rispettivi cataloghi configurati in [project_config.yml](project_config.yml).
Non si tratta di un catalogo globale in cui ogni paper compare una sola volta.

### `papers`: selezione e metadati originali

**Una riga per paper, per run.** Chiave logica: `(run_id, source, paper_id)`.

| Contenuto | Campi | Scopo |
|---|---|---|
| Identita' | `run_id`, `source`, `paper_id` | Identificare il paper nella specifica ricerca |
| Posizione nella selezione | `rank` | Conservare l'ordine dei risultati |
| Metadati bibliografici | `title`, `authors`, `abstract`, `published_at`, `updated_at` | Alimentare le note Obsidian e l'analisi bibliografica |
| Classificazione originale | `categories`, `primary_category` | Conservare i codici arXiv, come `cs.LG` |
| Riferimenti bibliografici | `doi`, `journal_ref` | Conservare identificatori e informazioni editoriali quando disponibili |
| Collegamenti | `html_url`, `pdf_url`, `pdf_volume_path` | Raggiungere la fonte e il PDF scaricato |
| Provenienza della ricerca | `search_topic`, `ranking_criteria`, `criteria_date`, `ingested_at` | Tracciare la ricerca e l'acquisizione che hanno prodotto la riga |

L'abstract originale rimane in questa tabella, distinto dal riassunto prodotto dall'LLM.

### `paper_enrichment`: informazioni derivate e citazioni

**Una riga di arricchimento per ciascun paper della run**, quando l'enrichment e' abilitato.
La chiave logica e' la stessa di `papers`: `(run_id, source, paper_id)`.

| Contenuto | Campi | Scopo |
|---|---|---|
| Identita' | `run_id`, `source`, `paper_id` | Collegarsi al paper corretto senza mescolare ricerche |
| Metadati per la visualizzazione | `title`, `authors` | Rendere titolo e autori disponibili anche nell'enrichment |
| Attribuzioni autore | `authorships` | Conservare nomi, posizione e ID autore restituiti dai provider, senza unirli per nome |
| Tag estratti dall'LLM | `tags` | Collegare paper semanticamente affini |
| Categorie arXiv umanizzate | `arxiv_category` | Mostrare categorie leggibili, distinte dai tag LLM |
| Riassunto LLM | `summary` | Fornire una sintesi distinta dall'abstract originale |
| Riferimenti recuperati | `bibliography` | Conservare tutti i riferimenti restituiti dal provider, inclusi quelli esterni ad arXiv |
| Stato del recupero | `citation_status`, `citation_error`, `citation_source`, `citation_fetched_at` | Distinguere recuperi riusciti, dati parziali, rate limit ed errori |
| Esiti dei provider | `citation_lookups` | Fonte, stato, errore e timestamp per ogni provider chiamato, compreso il primario fallito prima del fallback |
| Identita' del paper negli indici | `citation_paper_ids` | URL identificativi del paper citante nei provider risolti, utili per riconciliare nodi selezionati e riferimenti |
| Data arricchimento | `enriched_at` | Sapere quando e' stato prodotto l'enrichment |

Esempio: nella run selezioniamo A, B e C. Se il provider riporta che A cita B e un
paper esterno D:

```text
A
  bibliography: [riferimento a B, riferimento a D]
```

Il futuro grafo aperto potra' creare gli archi diretti `A -> B` e `A -> D`,
includendo D come nodo esterno. Una lista vuota con stato `rate_limited`
non significa "nessuna citazione": significa che il dato non e' stato ottenuto.
Anche un recupero `ok` dipende dalla copertura dell'indice bibliografico, non certifica
la completezza della bibliografia originale.

#### Contenuto di `bibliography`

La colonna e' un array di strutture: una voce per riferimento, nell'ordine restituito
dal provider (non necessariamente l'ordine della bibliografia originale per OpenAlex).
Non e' una nuova tabella e non aumenta `pipeline_runs.paper_count`: i paper selezionati
rimangono distinti dai riferimenti scoperti.

| Campo della voce | Contenuto |
|---|---|
| `reference_id`, `position`, `label` | ID della voce, posizione e label bibliografica quando presente; per OpenAlex la label non e' disponibile |
| `text` | Titolo/ID per OpenAlex; testo bibliografico normalizzato per il provider HTML |
| `urls` | Link HTTP/HTTPS presenti nella voce |
| `arxiv_ids`, `dois` | Identificatori espliciti normalizzati; conservati anche nelle voci ambigue |
| `target_id` | `scholar:<id>`, `openalex:<W-id>`, `arxiv:<id>`, `doi:<doi>` o `ref:<hash>` |
| `match_method` | `scholar_id`, `openalex_id`, `arxiv_id`, `doi` oppure `unresolved` |
| `evidence_url` | Lavoro citante nell'indice oppure URL HTML con anchor della voce |
| `provider_id` | URL identificativo del lavoro citato nel provider |
| `title`, `authors`, `publication_year` | Metadati del citato quando disponibili, null se non recuperati o in righe HTML/storiche |
| `authorships` | Autori strutturati del citato, con ID ed evidenza del provider quando disponibili |

Un riferimento non risolto resta consultabile ma non viene unito ad altre voci sulla sola
somiglianza del testo. I target OpenAlex/DOI/arXiv potranno diventare nodi condivisi del futuro grafo;
i placeholder non rappresentano paper globalmente disambiguati.

#### Identificativi degli autori

`paper_enrichment.authorships` descrive gli autori del paper selezionato;
`bibliography[].authorships` quelli del singolo lavoro citato. Ogni attribuzione contiene:

| Campo | Contenuto |
|---|---|
| `name`, `position` | Nome riportato dal provider e posizione a partire da 1 nella sua lista |
| `identifiers.semantic_scholar` | `authorId` Scholar, quando restituito |
| `identifiers.openalex` | URL identificativo autore OpenAlex, quando restituito |
| `identifiers.orcid` | ORCID restituito da OpenAlex, senza ricerca per nome |
| `provider` | `semantic_scholar` oppure `openalex` |
| `evidence_url` | URL del lavoro cui si riferisce l'attribuzione; per citati senza URL/ID, pagina del citante come evidenza bibliografica |

Questi dati provengono dalle chiamate ai paper gia' esistenti, anche per la bibliografia
paginata: nessuna chiamata API aggiuntiva per autore. `authors` rimane la lista dei nomi
per la visualizzazione; nei paper selezionati conserva i nomi originali della fonte.
Non si associano ID ai nomi di questa lista per posizione o somiglianza.

Se viene chiamato OpenAlex come fallback, le attribuzioni del paper recuperate da entrambi
i provider vengono conservate separatamente, anche quando il risultato bibliografico scelto
cambia. `provider` ed `evidence_url` indicano la provenienza di ogni attribuzione; `position`
e' locale alla lista di quel provider. Non e' una riconciliazione tra persone.
Gli autori omonimi rimangono distinti e gli ID assenti restano null.
`authorships` null significa dato non disponibile (anche HTML o righe storiche), mentre
`[]` rappresenta una lista esplicitamente vuota. Nessun backfill automatico.
Il futuro graph builder usera' questi identificativi senza assumere che un profilo
Scholar/OpenAlex sia una prova infallibile dell'identita' della persona.

Per consultare anche i riferimenti esterni:

```sql
SELECT
    paper_id AS citing_paper_id,
    reference.position,
    reference.text,
    reference.target_id,
    reference.match_method,
    reference.evidence_url
FROM llmops_dev.riccardo_tambone.paper_enrichment
LATERAL VIEW explode(bibliography) refs AS reference
WHERE run_id = '<RUN_ID>'
ORDER BY citing_paper_id, reference.position;
```

### `pipeline_runs`: registro e pubblicazione

**Una riga per run**, aggiornata durante l'esecuzione.

| Contenuto | Campi | Scopo |
|---|---|---|
| Identita' e tentativo | `run_id`, `attempt_id` | Identificare lo snapshot e il tentativo corrente |
| Parametri originali | `request_json`, `source`, `download_pdfs`, `enrich` | Conservare la richiesta e verificare che un retry usi gli stessi parametri |
| Stato generale | `status` | Distinguere `RUNNING`, `COMPLETED` e `FAILED` |
| Stati delle fasi | `ingestion_status`, `enrichment_status`, `export_status` | Tracciare avanzamento e fasi saltate o non implementate |
| Checkpoint selezione | `selection_complete` | Sapere se il retry deve riusare i paper gia' selezionati |
| Conteggi | `paper_count`, `pdf_count`, `enrichment_count` | Verificare consistenza e completezza del dataset |
| Diagnostica | `warnings`, `error` | Separare limitazioni recuperabili da errori della pipeline |
| Timestamp | `started_at`, `completed_at` | Tracciare inizio della run e fine del tentativo |
| Output futuri | `vault_path`, `zip_path` | Predisporre i percorsi dell'export, oggi non valorizzati |

```text
RUNNING
  |-- acquisizione, eventuali PDF/enrichment e verifiche riuscite
  |     `-- COMPLETED -> dataset leggibile
  `-- errore
        `-- FAILED -> dati eventualmente presenti, ma non pubblicati
```

Le scritture sulle tabelle sono transazioni separate: il registro e' una **barriera di
pubblicazione**, non una transazione unica su tutte le tabelle. L'export non e' ancora
implementato e il suo stato resta `NOT_IMPLEMENTED`.

### Volume: PDF per run

```text
/Volumes/<catalog>/<schema>/arxiv_files/
  runs/
    <run_id_1>/pdfs/...
    <run_id_2>/pdfs/...
```

`papers.pdf_volume_path` punta al documento scaricato. I PDF serviranno per l'export
Obsidian e per eventuali elaborazioni successive. Lo stesso paper in due ricerche puo'
avere due copie del PDF: non esiste ancora uno storage globale deduplicato.

### Storicizzazione: nuova ricerca e retry

Una nuova ricerca genera un nuovo `run_id`, anche ripetendo gli stessi parametri.
Lo stesso paper puo' appartenere a snapshot diversi senza essere un duplicato logico:

| run_id | source | paper_id | title |
|---|---|---|---|
| run_1 | arxiv | paper_A | Titolo A |
| run_1 | arxiv | paper_B | Titolo B |
| run_2 | arxiv | paper_A | Titolo A |
| run_2 | arxiv | paper_C | Titolo C |

La seconda ricerca non sovrascrive la prima. Un retry, invece, mantiene lo stesso
`run_id`, riusa selezione finalizzata e PDF presenti e aggiorna le stesse chiavi tramite
`MERGE`; l'enrichment puo' essere ricalcolato.

**Non viene conservata una cronologia applicativa completa dei tentativi**: il registro
mantiene il tentativo corrente e le righe della run vengono aggiornate. Lo storico qui
descritto riguarda run diverse, non ogni versione intermedia di un retry; non va confuso
con la cronologia tecnica delle transazioni Delta. Le run completate non sono modificabili
tramite i metodi di scrittura della pipeline. Non e' prevista una pulizia automatica dello storico.

### Letture corrette e scopo finale

I lettori richiedono un `run_id` esplicito, verificano che la run sia pubblicata, controllano
unicita', conteggi e corrispondenza delle chiavi e restituiscono i paper in ordine `rank`.
Non bisogna unire tutte le righe storiche solo su `paper_id`: si incrocerebbero snapshot diversi.
Questi controlli sono applicativi; Delta non impone una primary key univoca.

```text
papers + paper_enrichment della stessa run COMPLETED
                         |
                         v
               Formattazione delle note
                         |
             +-----------+-----------+
             v                       v
     Vault bibliografico       Vault tematico
     Paper e autori            Paper
     AUTHORS                   SHARES_TAG
     SHARED_AUTHORS            SHARES_CATEGORY
     CITES
```

Oggi sono implementati acquisizione, storage, enrichment e gestione delle run.
Il formatter Markdown, i grafi esportati, i vault e lo ZIP sono ancora da sviluppare:
le tabelle conservano i dati per costruirli, ma non viene ancora salvata una tabella
di nodi/archi ne' un vault pronto. Il contratto dell'output e' in
[PROJECT_PLAN.md](PROJECT_PLAN.md), gli schemi e i lettori in
[delta.py](src/autographrs/storage/delta.py).

### Run snapshots, retries and reads

Every new search gets a timestamp/topic/UUID `run_id`. The `pipeline_runs` Delta table
records the parameters (`request_json`, source and download/enrichment options), current
`attempt_id`, status, stage statuses, selection checkpoint, counts, warnings and errors.
Vault/ZIP paths and export status are reserved for the future exporter and remain unset.
The notebook prints the run ID and verifies the published dataset after ingestion.

- Leave the job's `run_id` parameter **empty** for a new search, even if its parameters
  match a previous search.
- To retry a **FAILED** run, supply its `run_id` with exactly the original search parameters.
  Running, completed and unknown run IDs are rejected.
- A retry reuses the finalized paper selection and existing PDF files. It upserts the
  enrichment again, so LLM/API calls may be repeated. If the selection failed before its
  checkpoint, only that run's unpublished paper rows are cleared and the search is repeated.
- `papers` and `paper_enrichment` use `MERGE` on `(run_id, source, paper_id)`, not blind append.
  Identical source duplicates are deduplicated; conflicting or historical duplicate keys
  fail explicitly. A finalized selection cannot be replaced with another paper set.
- `COMPLETED` is published only after both required stages and key/count checks succeed.
  Table writes are separate Delta transactions; the run registry is a publication barrier,
  **not** a multi-table transaction. Citation failures can produce a completed run with warnings.
  Missing requested PDFs fail the run rather than reporting download success.
- The bundle sets `max_concurrent_runs: 1`. Attempt checks additionally reject stale attempts.
  Do not run external/notebook writers concurrently against the same `run_id`; Delta does
  not enforce unique primary keys. A forcibly terminated job may leave `RUNNING`: inspect
  the job, confirm it is stopped and reconcile its registry state explicitly before retry.
  The pipeline does not automatically steal or expire active attempts.

Always use the scoped readers, never join entire historical tables by `paper_id` alone:

```python
from autographrs.config import load_config
from autographrs.storage.delta import DeltaPaperStore

store = DeltaPaperStore(load_config("project_config.yml", env="dev"))
run = store.load_run(run_id)
papers = store.load_papers(run_id)  # Ranked, published paper snapshot
enrichments = store.load_enrichments(run_id)  # Requires completed enrichment
dataset = store.load_run_dataset(run_id)  # List of (Paper, PaperEnrichment)
```

Readers reject unpublished runs, duplicate keys, inconsistent counts and missing/orphan
enrichments. An intentionally skipped enrichment is explicit, not an empty successful result.
Within a run, the combined reader matches `(source, paper_id)` and preserves paper rank.
Existing pre-registry rows remain untouched and are **not** implicitly published/backfilled.
The new enrichment `source` key column is added before merging when missing; legacy rows
retain a null source until deliberately migrated. Start a new run to test the new lifecycle.

### Small dev trial

```powershell
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

In the deployed **ingest_papers** job, choose **Run now with different parameters**:
set `topic` to your search, `num_papers` to `3`, `ranking_criteria` to `relevance`,
and leave `criteria_date`, `categories` and `run_id` empty for the first trial.
Keep `env=dev`. The final notebook cell verifies that published paper and enrichment
counts match; check `pipeline_runs.warnings` for external API limitations.

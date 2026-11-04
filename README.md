# llmops-riccardo-tambone

Your repository for the LLMOps course on Databricks. Everything you need in
the course workspace was provisioned for you before you got this repo, with its identifiers in
`project_config.yml`. AutoGraphRS uses the existing Lakebase compute with standard pgvector as described below.

## Join the course Discord

Join the course's Discord server with this invite link:
[https://discord.gg/V4XRYcBFDE](https://discord.gg/V4XRYcBFDE). Join under the
Discord username you wrote in the intake form: the course channels open automatically for members
on the student list.

## What was provisioned for you

| What | Value |
|---|---|
| Workspace, where you sign in | [https://dbc-e81ca4d2-f6bd.cloud.databricks.com](https://dbc-e81ca4d2-f6bd.cloud.databricks.com) |
| Your schema in development, which you manage | `llmops_dev.riccardo_tambone` |
| Your schema in acceptance, which you read | `llmops_acc.riccardo_tambone` |
| Your schema in production, which you read | `llmops_prd.riccardo_tambone` |
| Suggested volume name (create during the course) | `arxiv_files`, such as `llmops_dev.riccardo_tambone.arxiv_files` |
| SQL warehouse id | `cbb5a7d4b7979574` |
| Usage policy id | `891ab3b2-3b04-3ccb-8359-9327a16ef424` |
| MLflow experiments | Dev: `/llmops/riccardo-tambone-dev`; acc: `/llmops/riccardo-tambone-acc`; prd: `/llmops/riccardo-tambone-prd` |
| Chat model service | `course_ops.gateway.chat` |
| Embedding serving endpoint | `databricks-gte-large-en` |
| Embedding model service | `course_ops.gateway.embed` |

## Models: which name each call takes

`llm_endpoint` and `embedding_service` name model services of the course's AI Gateway, with a
monthly budget per student. Chat routes to GPT-OSS-120B; embeddings route to GTE Large. Call them
through the gateway with the `databricks-openai` package:

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

`embedding_endpoint` names the serving endpoint of the same GTE Large model, so its vectors match
the gateway's. Pass it wherever Databricks asks for a serving endpoint: to `ai_query` in SQL, as
an AI Search index's `embedding_model_endpoint_name`, and to `DatabricksServingEndpoint` among
an agent's resources. For example, in a notebook:

```python
spark.sql("SELECT ai_query('databricks-gte-large-en', text) AS embedding FROM chunks")
```

Neither name works in the other's place: the gateway client refuses the serving endpoint, and
`ai_query` refuses the model service. AutoGraphRS generates vectors through the AI Gateway
`embedding_service` (`course_ops.gateway.embed`) before storing them in Delta and indexing them.

## Lakebase

The `lakebase_instance` value in `project_config.yml` identifies your Lakebase project.
All three environments share it; your Unity Catalog schemas remain separate.
AutoGraphRS stores authoritative embeddings in Delta and publishes operational copies to
Lakebase PostgreSQL with the standard `vector` (pgvector) extension and cosine HNSW indexes.
The pipeline's `publish_search_index` task first prepares the extension and the environment's PostgreSQL
schema. It never enables **Lakebase Search**, creates computes or provisions AI Search.

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
Il graph builder legge la bibliografia per costruire archi `CITES`, compresi quelli
verso opere esterne alla selezione, in uno step separato.
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
  La bibliografia e' il dato di ingresso del grafo.
- Le categorie di altre fonti rimangono in `papers.categories`: non vengono convertite con
  la tassonomia arXiv. In quel caso `arxiv_category` e' vuoto e i tag LLM restano disponibili.

Il job cerca ancora su arXiv: il provider di ricerca PubMed non e' implementato.
La compatibilita' generica e' verificata con test di paper non-arXiv e persistenza end-to-end.
Ogni run usa una sola fonte di ricerca; non e' ancora una ricerca federata multi-repository.
Il provider HTML, arXiv-only, viene rifiutato esplicitamente su paper di altre fonti.

## Dati

Ogni ricerca produce un **dataset autonomo identificato da `run_id`**. AutoGraphRS salva
tre tabelle Delta di acquisizione/enrichment, due per gli abstract bibliografici e tre del grafo nello schema Unity
Catalog dell'ambiente selezionato e i PDF nel Volume
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

Il grafo aperto crea gli archi diretti `A -> B` e `A -> D`,
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
somiglianza del testo. I target Scholar/OpenAlex/DOI/arXiv possono diventare nodi condivisi del grafo;
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
          Graph builder -> nodi e archi CITES (implementato)
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

Oggi sono implementati acquisizione, storage, enrichment, gestione delle run e grafo delle
citazioni. Relazioni tematiche/autori, formatter Markdown, vault e ZIP restano da sviluppare.
Il contratto dell'output e' in
[PROJECT_PLAN.md](PROJECT_PLAN.md), gli schemi e i lettori in
[delta.py](src/autographrs/storage/delta.py).

## Graph builder: citazioni

Il primo incremento costruisce **nodi Paper e archi diretti `CITES`** senza chiamate API,
LLM, download o espansione ricorsiva. Input: `papers` e `paper_enrichment` della stessa run
con ingestion ed enrichment completati. La bibliografia rimane immutata.

Moduli in [graph/](src/autographrs/graph/):

- `models.py`: nodi, archi con evidenze, risultato e registro della build.
- `identity.py`: alias arXiv senza versione, DOI normalizzati e ID Scholar/OpenAlex.
- `builder.py`: funzione pura `build_citation_graph(run_id, dataset)`, testabile senza Spark.
- `storage.py`: `GraphStore`, pubblicazione e lettura Delta.

### Regole di costruzione

- Ogni paper selezionato rimane un nodo `selected`, anche senza riferimenti disponibili.
- Uno stesso lavoro esterno con identificativi compatibili diventa un nodo `external`
  condiviso. Alias acquisiti insieme permettono riconciliazioni transitive tra provider.
- Riferimenti senza identita' diventano nodi `unresolved`, locali alla singola occorrenza.
  Titoli o nomi simili non provocano fusioni.
- Una componente con ID diversi nello stesso namespace, oppure con piu' paper selezionati,
  e' ambigua: nessuna fusione nella componente, riferimenti locali e warning. Questa regola
  conservativa puo' mantenere separati duplicati reali, ma evita di scegliere un match arbitrario.
- Un solo arco per coppia citante/citato; tutte le occorrenze bibliografiche restano in
  `evidence` (paper citante, reference ID, posizione, testo, provider, URL).
  Nessun arco inverso inventato. I riferimenti al paper stesso sono omessi con warning.
- Bibliografie null o con stato diverso da `ok` producono warning di copertura; i dati
  disponibili vengono comunque usati. L'assenza di archi non certifica assenza di citazioni.

Gli ID e l'ordine del risultato sono deterministici per lo stesso input e versione delle
regole. Gli ID esterni possono cambiare se nuove evidenze modificano la riconciliazione:
non sono un registro globale permanente. Autori, `AUTHORS`, `SHARED_AUTHORS`, `SHARES_TAG`
e `SHARES_CATEGORY` non sono ancora costruiti; gli autori strutturati restano nell'enrichment.

### Tabelle Delta e pubblicazione

| Tabella | Chiave logica | Contenuto |
|---|---|---|
| `graph_nodes` | `(run_id, graph_build_id, node_id)` | Tipo `selected`/`external`/`unresolved`, stato identita', alias, titolo, autori, URL, abstract e `text_variant`; summary, tag, categorie e percorso PDF per i selezionati |
| `graph_edges` | `(run_id, graph_build_id, edge_id)` | `source_id`, `target_id`, relazione `CITES`, array di evidenze |
| `graph_builds` | `(run_id, graph_build_id)` | Versione builder, `bibliography_snapshot_id`, eventuali `embedding_snapshot_id` e `search_publication_id`, stato `RUNNING`/`COMPLETED`/`FAILED`, conteggi, warning, errore e timestamp |

In dev sono nello schema `llmops_dev.riccardo_tambone`.
Ogni esecuzione genera un nuovo `graph_build_id`: non modifica ingestion/enrichment o build
precedenti. Dopo le scritture, il builder rilegge e valida conteggi, contenuto ed estremi degli
archi, poi pubblica `COMPLETED`. Le transazioni Delta sono separate: il registro e' una
barriera applicativa di pubblicazione, non una transazione multi-tabella.
Build fallite o interrotte possono lasciare righe non pubblicate; niente pulizia automatica.

```python
from autographrs.config import load_config
from autographrs.graph import GraphStore

store = GraphStore(load_config("project_config.yml", env="dev"))
build = store.build_graph(run_id)  # Nessuna ripetizione delle chiamate ai provider
graph = store.load_graph(run_id, build.graph_build_id)
```

Il lettore richiede entrambi gli ID e rifiuta build non completate o strutturalmente
inconsistenti. Anche nelle query SQL filtrare sempre per `run_id` e `graph_build_id`
e controllare `graph_builds.status = 'COMPLETED'`, senza mescolare snapshot.

### Task del job Databricks

Il job si chiama **`job-autograph-pipeline`** (con il prefisso dev aggiunto dal bundle).
La definizione si trova in [autograph_pipeline.job.yml](resources/autograph_pipeline.job.yml).
La chiave interna del bundle rimane `ingest_papers` per aggiornare la risorsa esistente
senza crearne una nuova. Il job mantiene ambiente/versioni e aggiunge la dipendenza:

```text
ingest_papers -> enrich_bibliography -> prepare_embeddings -> publish_search_index -> build_graph
```

Il task ingestion pubblica il `run_id` effettivo con `dbutils.jobs.taskValues.set`;
i task dati successivi lo ricevono tramite `{{tasks.ingest_papers.values.run_id}}`,
senza un ID scelto dall'utente. Ogni avvio ordinario del job crea una nuova ricerca:
il primo task riceve `run_id: ""` e il sistema genera l'identificativo.
Il job non espone ne' `run_id` ne' `ingestion_run_id`.
Non aggiungere un parametro job `run_id`: Databricks lo propaga automaticamente e
gli da' precedenza sui parametri dei task, sovrascrivendo gli ID dei downstream.
Se il job e' stato configurato manualmente, rimuovere l'eventuale vecchio parametro
`run_id`/`ingestion_run_id` e aggiornare la definizione con il deploy.
Non introdurre questi ID negli override Run now/Repair a livello job.
Per recuperare un'ingestion FAILED, eseguire separatamente il notebook
[ingest_papers.py](resources/deployment_notebooks/ingest_papers.py) con il suo widget
`run_id` valorizzato all'ID esistente e gli stessi parametri originali.
Il Repair dei task successivi usa invece i task values gia' prodotti.
`enrich_bibliography` pubblica `bibliography_snapshot_id`, passato esplicitamente a
`prepare_embeddings` e `build_graph`. Il task embedding pubblica `embedding_snapshot_id`;
il task indice pubblica `search_publication_id`. Il grafo verifica che questi riferimenti
siano pubblicati e relativi alla stessa bibliografia, senza chiamate di rete, e registra
gli ID nella build. Pubblica poi `graph_build_id` per futuri consumer.

Se fallisce solo il grafo, usare **Repair run** sul task `build_graph`, evitando di
rieseguire ingestion su un `run_id` gia' completato. In alternativa eseguire il notebook
[build_graph.py](resources/deployment_notebooks/build_graph.py) con `env` e `run_id` espliciti.
Passare anche `bibliography_snapshot_id` per includere gli abstract esterni; lasciarlo vuoto
costruisce il grafo dalle sole bibliografie originali, senza nuove richieste di rete.
La run di acquisizione resta completata anche se il grafo fallisce; lo stato del job
Databricks e `graph_builds` segnalano il fallimento dello step.

## Embedding, indice e assistente bibliografico

Embedding e pubblicazione Lakebase/pgvector sono implementati e richiedono verifica sul workspace.
L'assistente read-only e il suo endpoint MCP sono implementati; il primo deployment App/Genie
e la verifica dei permessi sul workspace restano da eseguire. Il recupero degli abstract esterni e
la classificazione del testo sono implementati come descritto sotto. La pipeline batch
e il graph builder rimangono deterministici; un agente separato esplorera' la raccolta.
Obsidian restera' un output autonomo, utilizzabile senza agente o server MCP.

### Embedding e ricerca semantica

Implementazione in [embeddings/](src/autographrs/embeddings/), con gestione PostgreSQL in
[lakebase.py](src/autographrs/embeddings/lakebase.py). Si usa **Lakebase + pgvector standard**:
estensione `vector`, tipo `vector(n)`, indice HNSW con `vector_cosine_ops`.
**Non** si abilita Lakebase Search e non si usano `lakebase_vector`, `lakebase_ann` o AI Search.
SDK Databricks e `psycopg` sono gia' nelle dipendenze: nessun cambio delle versioni del corso.

1. `prepare_embeddings` carica la run completata e uno snapshot bibliografico esplicito.
   Riusa le stesse identita' e gli stessi metadati del graph builder, ma non pubblica
   un grafo intermedio. Include selezionati e citati esterni; riferimenti locali con
   un vero titolo possono essere incorporati senza diventare identita' riconciliate.
2. Testo: titolo ripulito ai margini, seguito da due newline e abstract se disponibile.
   `title_only` rimane distinto da `title_abstract`. Nessun PDF, chunk, summary o
   identificatore usato al posto del titolo. Nodi senza titolo utilizzabile esclusi
   con conteggio e warning, non eliminati dal grafo.
3. Chiama `course_ops.gateway.embed` sull'AI Gateway `/ai-gateway/mlflow/v1`, con auth
   Databricks del contesto, batch di default 32, timeout di 120 secondi per richiesta
   e massimo 2 retry del client. Errori provider fanno fallire lo step, senza vettori fittizi.
   Indici di risposta, dimensioni, valori finiti e vettori non nulli sono validati.
   `embedding_max_chars` (default 24000) e' una guardia in caratteri, **non** il limite
   token del modello: nessun troncamento silenzioso; input rifiutati dal provider
   devono essere gestiti esplicitamente.
4. Salva vettori float32 e un nuovo snapshot immutabile. Riusa contenuti identici
   **entro la stessa run**, anche dai batch salvati prima di un fallimento;
   nessuna cache globale tra run in questa iterazione. Testo/modello differenti non
   riusano lo stesso embedding. La dimensione viene ricavata dalla risposta, non presunta.
5. `publish_search_index` chiama il servizio `SearchStore.publish()`, che prima verifica
   l'estensione standard `vector` e la installa se assente, preparando schema PostgreSQL e
   registro dell'ambiente nel progetto esistente. Il setup e' idempotente e automatico
   anche nelle chiamate Python dirette; snapshot vuoti non richiedono accesso PostgreSQL.
6. Lo stesso servizio scrive direttamente in PostgreSQL, **senza synced tables o CDF**.
   Una transazione crea la tabella dello snapshot, inserisce i documenti senza sovrascrivere
   quelli esistenti, rilegge e confronta tutti i metadati e i vettori float32 con Delta,
   crea/verifica HNSW e registra un fingerprint completo. Una query di prova precede il commit.
   Solo dopo il commit PostgreSQL viene registrato `COMPLETED` in Delta.
   Uno snapshot senza documenti e' pubblicato solo in Delta, senza tabella/indice PostgreSQL.

| Tabella/risorsa | Contenuto |
|---|---|
| `paper_embeddings_<model_key16>` | Testo, titolo, tipo nodo, variante, vettore, dimensione, hash, endpoint/revisione, modello dichiarato nella risposta e chiavi run/snapshot/nodo |
| `embedding_snapshots` | Registro `RUNNING`/`COMPLETED`/`FAILED`, bibliografia di origine, versione identita', conteggi, warning ed errori |
| PostgreSQL `autographrs_<env>.e_<model_key16>_<snapshot_id>` | Copia operativa di un singolo snapshot, vettori `vector(n)` e indice `<tabella>_hnsw` |
| PostgreSQL `autographrs_<env>.embedding_publications` | Marker transazionale con modello, dimensione, tabella, conteggi e fingerprint dell'intero snapshot |
| Delta `search_publications` | Tentativi, backend `lakebase_pgvector`, host/database/tabella/indice, snapshot, conteggi, stato ed errori |

`model_key` deriva da endpoint, `embedding_revision` e versione di preparazione testo.
**`embedding_revision: v1` e' un'etichetta gestita da noi, non un pin del modello sottostante.**
Quando cambia il routing/modello del Gateway va incrementata; si ottengono nuova tabella
e nuovi indici per i nuovi snapshot. Il campo `response_model` aiuta a rilevare cambi, ma non puo' dimostrare
che un alias del Gateway non sia cambiato. Non mescolare spazi vettoriali.
Non esiste una transazione distribuita PostgreSQL/Delta: se Delta fallisce dopo il commit,
il retry rilegge e valida i dati gia' presenti, riusa l'indice e completa una nuova
pubblicazione Delta, **senza rigenerare embedding**. Advisory lock transazionali serializzano
il setup e i publish concorrenti dello stesso snapshot. Dati incompatibili causano errore,
non vengono sovrascritti. Nessuna eliminazione automatica di snapshot o indici storici.

Ogni snapshot ha una propria tabella/indice, anziche' un HNSW condiviso con filtri:
i filtri applicati dopo una ricerca ANN possono restituire meno di k risultati.
Qui il candidato ANN appartiene gia' allo snapshot richiesto. Per tabelle piccole
PostgreSQL puo' comunque preferire una scansione esatta: l'indice esiste ma non si forza
un piano inefficiente. Piu' snapshot comportano piu' oggetti/storage e richiederanno una
retention esplicita. Il limite corrente e' **1–2000 dimensioni**, compatibile con HNSW
su `vector`; dimensioni superiori sono rifiutate, senza conversioni silenziose a `halfvec`.

#### Setup Lakebase condiviso e autenticazione

In [project_config.yml](project_config.yml) tutti gli ambienti usano il progetto
`lb-llmops-riccardo-tambone`, l'host diretto gia' esistente e `databricks_postgres`.
I dati PostgreSQL sono separati in `autographrs_dev`, `autographrs_acc`, `autographrs_prd`;
quelli Delta restano nei rispettivi cataloghi UC. Capacita', costi, storage e disponibilita'
del progetto sono condivisi; gli schemi da soli **non** sono un confine di autorizzazione.
Configurare grant per le identita' dei tre ambienti secondo i permessi desiderati.
Il task [publish_search_index.py](resources/deployment_notebooks/publish_search_index.py)
include il setup nel servizio applicativo: nessun task/job/notebook di setup separato.
`LakebasePgVectorManager.setup()` e `.publish()` mantengono responsabilita' e transazioni
distinte, usando lo stesso manager. Errori di setup segnano il tentativo come FAILED e
impediscono la scrittura dello snapshot; Repair ripete il setup idempotente senza embedding.

Il manager risolve programmaticamente il compute read/write cercando l'**host esatto**
tra gli endpoint del progetto: non presume che il nome visualizzato della branch
`produzione` sia il suo ID tecnico. In alternativa impostare `lakebase_endpoint` al resource
path completo `projects/.../branches/.../endpoints/...`, comunque verificato rispetto a
progetto e host configurati. Un host assente o ambiguo produce errore esplicito.

Ogni connessione usa un nuovo token OAuth da
`WorkspaceClient.postgres.generate_database_credential`, TLS (`sslmode=require`),
timeout di connessione 10s, lock 30s e statement `search_timeout_seconds` (default 1800s).
La connessione diretta viene chiusa al termine, favorendo lo scale-to-zero esistente;
non si usa il pooler PgBouncer. Fino a 3 tentativi per errori transitori riconosciuti
(SQLSTATE classe `08` o `57P03`), non per autorizzazione, errori non classificabili
o per il replay automatico di transazioni.
Il ruolo PostgreSQL deriva dall'identita' SDK, oppure da `lakebase_user` esplicito.
**Nessuna password/token nel YAML.**

```powershell
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev ingest_papers
```

Il deploy aggiorna il job; l'esecuzione usa/risveglia **compute Lakebase a pagamento gia'
esistente**, senza creare progetti, branch, endpoint o modificare scale-to-zero e policy.
Prerequisiti: accesso di rete dal job serverless all'host diretto; accesso SDK al progetto
e generazione credenziali; ruolo OAuth PostgreSQL della **Run as identity**, con CONNECT,
diritti per installare `vector` e creare schema/tabelle/indici. Se il ruolo non puo' installare
estensioni, far installare `vector` da un amministratore. Verificare anche i permessi UC/Gateway.
I service principal acc/prd devono avere ruoli/grant propri: i grant UC non valgono in PostgreSQL.
Disponibilita'/versione di `vector`, permessi e connessione vanno verificati sul Lakebase reale.

**Migrazione da AI Search:** gli embedding Delta esistenti sono riutilizzabili. Dopo il
deploy usare Repair includendo `publish_search_index` e `build_graph`.
Le vecchie pubblicazioni Vector Search non diventano pubblicazioni pgvector: ripubblicare
lo snapshot. Non vengono cancellati vecchi endpoint/indici AI Search: dismetterli
esplicitamente solo dopo la verifica della nuova pipeline, per evitare costi residui.

#### Ricerca applicativa (base del futuro tool)

```python
from autographrs.config import load_config
from autographrs.embeddings.search import SearchStore

search = SearchStore(load_config("project_config.yml", env="dev"))
results = search.search(
    "recognition memory in perirhinal cortex",
    run_id=run_id,
    snapshot_id=embedding_snapshot_id,
    publication_id=search_publication_id,
    limit=5,
)
```

Il servizio richiede una pubblicazione COMPLETED, incorpora la query con lo stesso
Gateway/revisione e verifica modello/dimensione. Verifica anche il marker PostgreSQL
contro lo snapshot Delta, poi interroga la sola tabella di quello snapshot usando
la distanza coseno `<=>`. Restituisce una lista di dizionari con identita'/testo/provenienza
(senza il vettore) e `cosine_similarity = 1 - distanza`, **non una probabilita'**.
I filtri applicativi non sostituiscono ACL e autorizzazioni: il futuro agente dovra'
verificare quali snapshot l'utente puo' interrogare. Non e' ancora un agente o una
risposta RAG generativa. Non usiamo SQL `ai_search()`: PostgreSQL riceve il vettore generato
dal Gateway. Filtri per anno e ranking ibrido lessicale/vettoriale dello snippet del corso
non sono ancora implementati; la API applicativa richiede sempre run, snapshot e pubblicazione.

**Nessun arco `SIMILAR_TO` ancora generato.** Sara' un incremento separato con top-k,
soglia e metrica valutati; la similarita' non prova una citazione o l'identita' di due lavori.
Per Repair, ripetere `publish_search_index` se fallisce solo la pubblicazione PostgreSQL/Delta:
non richiama il modello. Se si ripete `prepare_embeddings`, rieseguire anche i downstream
per propagare il nuovo snapshot. Un errore dell'indice blocca il grafo nel job, ma non
invalida la run di ingestion o gli embedding gia' completati.

### Abstract dei citati esterni

Implementato nel task [enrich_bibliography.py](resources/deployment_notebooks/enrich_bibliography.py),
con resolver [abstracts.py](src/autographrs/enrichment/abstracts.py) e persistenza
[bibliography.py](src/autographrs/enrichment/bibliography.py).
L'ingestion continua a produrre la bibliografia originale, senza sovrascriverla.
Lo step separato:

1. Riconcilia gli ID con le stesse regole conservative del grafo; ogni nodo esterno
   viene interrogato una volta per provider nella stessa esecuzione. Riferimenti a paper
   selezionati usano gia' il loro abstract originale e non richiedono altre chiamate.
2. Scholar: batch fino a 100 lavori con `paperId,title,abstract`, identificati per
   ID Scholar, DOI o ID arXiv. Nessuna bibliografia aggiuntiva richiesta.
3. Per abstract mancanti: OpenAlex in batch fino a 100 ID OpenAlex o DOI esatti.
   `abstract_inverted_index` viene ricostruito per posizione, rifiutando indici malformati.
   Senza un ID OpenAlex/DOI non viene fatta ricerca per titolo.
   Per abstract ancora mancanti, terzo fallback all'API arXiv con ID arXiv esatti,
   batch fino a 100, pacing 3 secondi e 2 retry del client esistente. Versioni arXiv
   normalizzate; ID estranei/duplicati rifiutati, nessun matching per titolo.
   Un errore esaurito interrompe i batch arXiv restanti con diagnostica esplicita.
4. Riusa pacing, autenticazione e retry dei provider. Scholar condivide il budget retry
   di 90 secondi tra i batch; dopo esaurimento si saltano i batch restanti.
   OpenAlex interrompe i batch della stessa famiglia di ID dopo un errore esaurito.
   Non e' un limite rigido alla durata totale del task: pacing/timeout dipendono dai lavori.
5. Salva uno snapshot immutabile con titolo, abstract, provider, URL, timestamp ed esiti
   separati di ogni tentativo provider. Non modifica snapshot di ingestion o grafi precedenti.

Prima dei lookup consulta `bibliography_abstract_cache`, cache Delta persistente nello
schema dell'ambiente e condivisa tra run/Repair. Riusa **solo titolo+abstract recuperati
con successo**; non memorizza assenze, title-only, errori o rate limit. TTL predefinito
30 giorni, configurabile con `bibliography_cache_days` (`0` disabilita).
La chiave usa l'intero insieme di alias esatti riconciliati e la versione delle regole
di identita': nessuna unione per titolo o per un singolo alias in conflitto.
Alias aggiunti/modificati provocano un cache miss conservativo; nodi unresolved esclusi.
Gli hit conservano provider, URL, data originale e tentativi, rimappando soltanto
il node_id della run corrente. I successi vengono salvati prima dello snapshot, cosi'
una successiva pubblicazione fallita non obbliga a ripetere i lookup gia' riusciti.
La cache e' aggiornabile, gli snapshot rimangono immutabili; letture/scritture fallite
sono errori espliciti, non fallback silenziosi. Record scaduti vengono ignorati e
aggiornati al prossimo successo, senza cancellazione automatica.

| Tabella | Chiave | Contenuto |
|---|---|---|
| `bibliography_metadata` | `(run_id, snapshot_id, node_id)` | Titolo, abstract, `text_variant`, stato, provenienza e diagnostica |
| `bibliography_snapshots` | `(run_id, snapshot_id)` | Versione identita', stato pubblicazione, conteggi, warning, errore e timestamp |

`text_variant` e' **`title_abstract`** se titolo e abstract sono disponibili,
**`title_only`** se e' disponibile soltanto il titolo, **null** senza un titolo utilizzabile.
Un identificatore o una citazione grezza non viene spacciata per titolo.
Le etichette sono salvate nei metadati e sui nodi del grafo, anche selezionati;
il task successivo `prepare_embeddings` usa questa classificazione.

Stati: `ok`, `missing` (abstract non restituito), `unresolved` (identita' incerta),
`rate_limited`, `error`. Errori provider possono produrre snapshot completati con warning
e dati title-only, senza nascondere l'errore negli `attempts`. Errori di scrittura/consistenza
producono snapshot `FAILED`, non leggibili. Ogni riesecuzione crea un nuovo snapshot;
Repair del task puo' ripetere chiamate ai provider, senza danneggiare lo storico.
Il confronto di rilettura normalizza in UTC i timestamp restituiti da Spark,
anche nei tentativi provider annidati: perdere l'informazione di timezone nel
roundtrip TIMESTAMP non e' una modifica dei dati. Differenze effettive continuano
a bloccare la pubblicazione, con ID nodo e nomi dei campi discordanti nell'errore.

```python
from autographrs.enrichment.bibliography import BibliographyStore
from autographrs.graph import GraphStore

metadata_store = BibliographyStore(config)
snapshot = metadata_store.enrich_bibliography(run_id)
metadata = metadata_store.load_metadata(run_id, snapshot.snapshot_id)
build = GraphStore(config).build_graph(run_id, bibliography_snapshot_id=snapshot.snapshot_id)
```

Il builder legge lo snapshot indicato, ne verifica versione, conteggi e corrispondenza
dei nodi, e registra `bibliography_snapshot_id` nella build. Recuperare la provenienza
con una join su run, snapshot e node_id, mai su un generico titolo.

Questo arricchisce solo i citati a **un livello**, senza recuperarne automaticamente la
bibliografia, scaricare PDF o eseguire sintesi LLM. I placeholder senza ID affidabile
rimangono esclusi dai lookup automatici per titolo. Limiti di copertura, licenze e API
rimangono applicabili; l'assenza dell'abstract non elimina il nodo o la citazione.
Il builder consuma lo snapshot esterno esplicito, senza fare chiamate di rete.

### Assistente bibliografico: App privata, pgvector e Genie Agent

Implementazione in [assistant/](src/autographrs/assistant/). L'assistente e' **read-only**,
separato dalla pipeline batch. Il codice gira in una **Databricks App**, usa il modello chat
del Gateway del corso per il tool calling e affianca un **Genie Agent** (ex Genie Space)
per analisi SQL sulle viste curate. Nessun nuovo endpoint Model Serving, AI Search,
cluster Spark, progetto o compute Lakebase.

```text
Utente autenticato -> Databricks App -> Gateway chat del corso
                         |-> search_collection -> Gateway embedding + Lakebase/pgvector
                         |-> get_paper / get_bibliography / get_neighbors -> SQL Warehouse
                         |-> ask_collection_analytics -> MCP gestito Genie -> SQL Warehouse
Client MCP autenticato -> stessa App /mcp -> stessi servizi Python
```

Il setup approva **una sola raccolta per App** tramite `run_id` + `graph_build_id` espliciti.
Verifica grafo, bibliografia, pubblicazione embedding e contenuti coerenti, poi prepara
un manifest non segreto in Delta `assistant_settings`. La App lo carica alla prima
richiesta autenticata con consenso OAuth, e lo mantiene fino al riavvio:
il modello e il client non possono scegliere run, snapshot, tabelle o endpoint differenti.
Nessuna selezione implicita della run piu' recente e nessun caricamento dell'intero dataset
con Spark nel percorso interattivo.

| Tool implementato | Responsabilita' |
|---|---|
| `search_collection` | Top-k semantici di tutti i tipi di nodo, con `kind`, variante testo e anteprima (800 caratteri, troncamento esplicito); verifica marker/fingerprint PostgreSQL, modello e dimensione; non e' un filtro tematico esaustivo |
| `get_paper` | Identificatori, titolo, abstract e metadati del nodo esatto |
| `get_bibliography` | Riferimenti originali acquisiti, anche irrisolti, con stato/provenienza provider e paginazione; bibliografie esterne non inventate |
| `get_neighbors` | Navigazione locale CITES, massimo due hop e default/massimo 100 vicini; nodi compatti (ID, tipo, titolo, anno, disponibilita' abstract), archi con evidenze; `get_paper` per il dettaglio |
| `ask_collection_analytics` | Conteggi, filtri tematici sulle viste e ranking delle citazioni locali via Genie; non usa solo il top-k semantico |

`search_sources` e `propose_expansion` restano futuri: nessuna acquisizione o modifica automatica.
La risposta chat include `answer`, `evidence` (risultati reali dei tool), `tool_calls` e `tokens`.
La UI mostra le evidenze espandibili come testo, senza interpretare HTML restituito dal modello.
Riferimenti `[node_id]`, titoli e URL devono essere verificati nelle evidenze; il prompt non
costituisce una garanzia matematica di correttezza delle risposte.
`selected` identifica i paper acquisiti, `external` lavori esterni identificati e
`unresolved` riferimenti non risolti: non sono tre livelli equivalenti di copertura.
L'assenza di bibliografia in un nodo esterno non significa che il lavoro non abbia
citazioni. `outgoing` segue i riferimenti del paper; `incoming` trova chi lo cita
nel grafo. I dati dei nodi non contengono un conteggio globale aggiornato delle citazioni.
Per "paper EEG piu' citato", Genie deve filtrare i campi disponibili della raccolta
e contare CITES entranti; se si chiedono citazioni globali, l'agente deve chiarire il limite.

#### Risorse e autorizzazioni

[bibliographic_assistant.yml](resources/bibliographic_assistant.yml) aggiunge al Bundle:

- App `autographrs-rtambone-<target>` (23 caratteri per dev/acc/prd, limite API 30),
  privata, con il solo deployer `CAN_MANAGE`;
  nessuna condivisione con `users` o `All account users`. La App ha un proprio service principal.
- Job separato `setup_assistant`, runtime batch **5** invariato, eseguito esplicitamente
  dal proprietario per approvare una raccolta. Non fa parte dell'ingestion.
- Uso del Warehouse esistente con **User Authorization**: scope OAuth `sql` e `genie`;
  anche chat/embedding usano l'utente con scope `ai-gateway`.
  Nessuna risorsa Warehouse allegata alla App, nessun CAN_MANAGE richiesto al deployer
  sul Warehouse condiviso.

Il job prepara viste UC `assistant_<hash>_papers` (solo `kind='selected'`), `_nodes` (tutti i
nodi del grafo), `_citations`, `_bibliography`, vincolate
alla build approvata. Il suffisso dipende anche dalla build: cambiare raccolta non
riscrive una vista usata dal processo precedente. Il proprietario accede a queste viste
con i propri permessi UC e `CAN_USE` sul Warehouse. Il setup predefinito `sql_auth=user`
non assegna grant UC al service principal della App; passando da un precedente manifest
`sql_auth=app` revoca i SELECT dedicati di quella configurazione.

Il job crea/aggiorna **Genie via SDK**, con istruzioni ed esempi SQL versionati in
[bootstrap.py](src/autographrs/assistant/bootstrap.py), dopo la creazione delle viste.
La vista `_papers` contiene **solo i paper acquisiti**, quindi "quanti paper" non puo'
contare i lavori citati; `_nodes` contiene tutti i nodi con `kind` (`selected` = paper
acquisiti, `external` = lavori citati non acquisiti, `unresolved` = riferimenti non risolti).
Le istruzioni spiegano che "piu' citato" conta i CITES in ingresso dai paper selezionati su
qualsiasi tipo di nodo, con esempi per conteggio paper, conteggio per tipo, ranking globale
e ranking filtrato per argomento (titolo/abstract/tag).
I conteggi di citazioni **globali** non sono acquisiti: se l'utente chiede "il piu' citato"
senza qualificarlo, l'agente dichiara che il dato globale non e' disponibile e restituisce il
ranking locale etichettato "nella raccolta"; "piu' citato nella raccolta" restituisce
direttamente il ranking locale.
Modifiche a istruzioni/esempi richiedono di rieseguire `setup_assistant`.
Gli identificatori delle tabelle e gli esempi nel payload sono ordinati lessicograficamente,
come richiesto dal formato export proto di Genie.
Genie non e' una risorsa nativa `genie_spaces` nel YAML: questo ordine evita di creare
uno spazio riferito a viste inesistenti al primo deploy. L'ID viene mantenuto nel manifest
e pubblicato come task value; i retry riusano lo spazio. Il Bundle gestisce il job che
ne governa la configurazione, non una seconda definizione concorrente dello spazio.
Lo spazio cura solo `_papers`, `_nodes` e `_citations`; in modalita' user la ACL consente il
proprietario, non concede CAN_RUN alla App. Il tool Genie usa l'OAuth dell'utente.
Genie genera SQL nel suo percorso autorizzato: il divieto di SQL libero vale per
l'orchestratore/tool personalizzati, non per il funzionamento interno di Genie.

In Lakebase il setup crea, se manca, il ruolo OAuth del service principal tramite
`databricks_create_role(..., 'SERVICE_PRINCIPAL')`, dopo aver inizializzato
`CREATE EXTENSION IF NOT EXISTS databricks_auth` nel database di connessione,
e concede CONNECT, USAGE sullo schema
e SELECT sul registro pubblicazioni e sulla sola tabella vettoriale approvata.
La funzione e' fornita da tale estensione: se manca, PostgreSQL restituisce
`UndefinedFunction`. Il setup richiede quindi anche il permesso di installare
`databricks_auth` se non e' gia' presente; un diniego resta un errore esplicito.
La App usa transazioni PostgreSQL read-only e non invoca setup/publish.
Una raccolta senza documenti indicizzabili non apre connessioni PostgreSQL e restituisce
zero risultati semantici; metadati, grafo e Genie restano disponibili. Passando da una
raccolta non vuota a una vuota, il setup revoca comunque l'accesso alla vecchia tabella.
Il job rifiuta ruoli amministrativi, privilegi CREATE ereditati o scrittura sulle tabelle
di retrieval; non modifica grant generali del corso per aggirare questi controlli.
Alla riconfigurazione revoca i grant dedicati alle vecchie viste e alla vecchia tabella
vettoriale, senza eliminare dati. Fermare la App prima di cambiare raccolta.

Il setup controlla anche le ACL effettive App/Genie e fallisce se rileva condivisioni
dirette/ereditate estranee al proprietario, all'App o al gruppo amministratori.
Il servizio richiede `X-Forwarded-User` e confronta gli identificatori inoltrati
(`X-Forwarded-User`, `X-Forwarded-Email`, `X-Forwarded-Preferred-Username`)
con l'allowlist ID/username del proprietario registrata dal setup.
L'identificatore IdP in `X-Forwarded-User` puo' differire dall'ID SCIM Databricks:
email/username permettono di riconoscere lo stesso proprietario senza aprire l'accesso ad altri.
Questi header sono attendibili
**solo dietro il proxy autenticato di Databricks Apps**: non esporre questo processo
come server pubblico autonomo fidandosi di header forniti dal client.

Identita' distinte, senza fallback automatico:

| Servizio | Credenziale runtime |
|---|---|
| Lettura manifest, metadati, bibliografia, grafo via SQL | Token OAuth dell'utente inoltrato dal proxy Apps |
| Genie MCP | Stesso token utente, scope `genie` |
| Gateway chat ed embedding | Stesso token utente, scope `ai-gateway` |
| Lakebase/pgvector | OAuth del service principal App, ruolo/grant esistenti |

Il proxy fornisce `X-Forwarded-Access-Token` dopo consenso agli scope `sql`/`genie`/`ai-gateway`.
Il token e il relativo client sono isolati nella singola richiesta, anche nei worker
dei tool, e non vengono salvati in manifest, file, log o cache globale. Non si usa
un PAT personale configurato come segreto App. Un token mancante restituisce 401;
un errore SQL/Gateway o un accesso negato non fanno ripiegare sull'identita' della App.
Anche i client MCP devono ottenere dal proxy l'identita' e il token delegato:
conoscere l'URL o usare la sola identita' tecnica non sostituisce il consenso.

Rimosso il blocco risorsa Warehouse che richiedeva CAN_MANAGE al deployer.
Questo risolve quel requisito, **non** concede nuovi permessi: sul Gateway valgono
i grant e budget dell'utente, inclusi quelli ereditati dal gruppo studenti.
Lakebase continua invece a richiedere grant e accesso API/rete per il service principal
App: il successo della pipeline personale non prova che questa identita' sia autorizzata.
Eventuali dinieghi restano espliciti. User Authorization e tutti e tre gli scope
devono essere consentiti dalle policy del workspace.

La documentazione Apps include `ai-gateway` tra gli
[scope supportati](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth#supported-scopes).
Per aggiornare un'App gia' configurata, eseguire `databricks bundle deploy -t dev`,
poi `databricks bundle run -t dev bibliographic_assistant` e riaprire l'URL per accettare
il nuovo consenso. Non serve rieseguire ingestion o setup: il manifest non cambia.
Non basta Start, che riusa il codice gia' distribuito. Verificare sul workspace una
domanda chat e una ricerca semantica; i test locali non certificano token/scope reali.

Gli errori HTTP del client OpenAI/Gateway restituiscono 502 con un reference locale
e diagnostica controllata visibile anche nella chat: status HTTP upstream,
operazione chat/embedding, code/type/param e request ID se presenti e validi.
Dal messaggio provider si estraggono solo nomi di campi della richiesta riconosciuti
(`mentioned_fields`): una menzione non dimostra che quel campo sia la causa.
Gli stessi metadati sono nei log con il reference; non vengono stampati il corpo
grezzo dell'errore, prompt, risposte, header o token. Per un HTTP 400 copiare questa
diagnostica prima di modificare parametri o autenticazione; il reference da solo
non identifica la causa e un nuovo access token non corregge un parametro non supportato.
Il client chat omette il parametro opzionale `parallel_tool_calls`, menzionato dal
Gateway del corso in una risposta HTTP 400. Eventuali chiamate multiple restituite
dal modello vengono comunque eseguite in sequenza dal codice, entro i budget esistenti.
Gli errori `ExceptionGroup` includono `Group diagnostics`: tipi delle eccezioni interne,
modulo applicativo di origine se disponibile e status HTTP per errori di trasporto.
Le chiamate HTTP al percorso Genie MCP sono etichettate `genie_mcp`; eventuali errori
Gateway interni conservano la diagnostica controllata descritta sopra. L'ispezione
e' limitata a 64 elementi/16 cause, con flag `truncated`, senza messaggi grezzi,
URL completi, header, corpi delle risposte o variabili locali.
Il rifiuto del contratto Genie riporta un `GenieContractError` con motivo
`ambiguous_question_tools` o `unsupported_question_schema` e soli conteggi/tipi
degli argomenti MCP, anche dentro un gruppo. Nessuna domanda viene inviata a un
tool con contratto non riconosciuto.
Il server puo' esporre piu' tool: l'adattatore seleziona l'unico con un solo argomento
obbligatorio di tipo stringa, lasciando omessi gli argomenti opzionali.
Tool con due argomenti obbligatori non vengono invocati; zero candidati o piu'
candidati compatibili causano un errore esplicito, indipendentemente dall'ordine.
La validazione distingue richieste chat non valide (400) da argomenti tool non validi
generati dal modello (502 con `Tool diagnostics`: nome tool, campo e tipo di vincolo,
senza valori). Errori di validazione dei dati restituiti sono 502 con reference.
I limiti e il rifiuto di argomenti extra restano invariati: nessun troncamento o
conversione automatica dei parametri per nascondere una chiamata non valida.
Nel loop chat un argomento tool non valido produce un feedback strutturato al modello,
senza eseguire il tool. Lo stesso vale per gli errori recuperabili (`RecoverableToolError`):
`node_not_found` (node_id inesistente o inventato), `bibliography_not_published`,
`tool_response_too_large`, `unknown_tool` e JSON di argomenti non valido. Sono consentite
al massimo tre correzioni per richiesta; l'errore successivo si propaga con il suo codice.
Anche i tentativi rifiutati contano nel budget di sei
chiamate, entro cinque round/120s, e sono riportati nelle evidenze come errori,
non dati della raccolta. Gli errori di servizio (SQL, Gateway, Genie) non attivano questa correzione.
Se il modello chiude un turno senza testo e senza tool call, oppure quando i budget impongono
la risposta, l'agent esegue un round di **sintesi senza schemi tool**: le evidenze raccolte
vengono riproposte come dati delimitati (`<evidence>`) in un messaggio utente, cosi' il modello
non puo' tentare chiamate tool soppresse (osservato con il Gateway del corso: `finish_reason=stop`
e contenuto vuoto con `tool_choice="none"`). Se anche la sintesi e' vuota l'errore
`gateway_incomplete_answer` riporta `finish_reason`, round, `content_chars` e `retried`.
Il risultato di Genie e' compattato in `text`, `queries` (SQL, colonne, massimo 30 righe,
`row_count`, `truncated`), senza la doppia copia testo/structuredContent e i metadati statement.
Genie MCP e' asincrono: se il tool di domanda restituisce uno stato non terminale
(es. `EXECUTING_QUERY`), il client interroga il tool di stato (argomenti
`conversation_id`/`message_id` riconosciuti per nome) con backoff fino a `COMPLETED`, entro
i 40s complessivi. `FAILED`/`CANCELLED` diventano `genie_query_failed`; il timeout diventa
`genie_timeout`, restituito al modello come errore correggibile.

**Stile delle risposte.** Il prompt separa le regole interne dallo stile verso l'utente:
risposte dirette e brevi in Markdown leggero, lavori citati per titolo/autori/anno con il
node_id tra parentesi (per domande successive), senza nomi dei tool o delle fonti (mai
"(Genie)"), punteggi di similarita' o sezioni "Fonti";
le evidenze grezze restano nel blocco richiudibile "Dettagli tecnici". I limiti sono
menzionati solo quando cambiano la conclusione.

**Interfaccia.** La UI (`assistant/ui.py`, nome visualizzato "AutoGraph") e' una chat a
finestra unica con schermata di benvenuto e domande d'esempio. Il Markdown e' reso da un
piccolo renderer interno che costruisce nodi DOM (niente `innerHTML`, niente CDN, compatibile
con la CSP `script-src 'self'`); i link sono solo http(s) e si aprono in una nuova scheda.

**Log dell'agent.** Ogni richiesta ha un `trace_id` (mostrato in "Dettagli tecnici" sotto la risposta
e nei diagnostics degli errori). L'agent scrive una riga JSON `agent_trace` per ogni evento
(`model_round`, `tool_call`, `empty_answer_retry`, `request_completed`, `request_failed`) con
solo metadati: round, `finish_reason`, token prompt/completion, tool richiesti, esito e durata
in ms. Mai prompt, risposte o payload dei tool. Si leggono in Databricks dalla pagina
dell'App → scheda **Logs**, filtrando per `agent_trace` o per il `trace_id`.
Il tracing MLflow (span per round/tool) e' previsto come estensione successiva e riusera'
gli stessi punti di misura.
Il prompt indirizza ranking/conteggi completi delle citazioni a Genie: una pagina
`get_neighbors` (default e massimo 100 vicini, limiti inferiori rispettati) non dimostra
un ranking globale della raccolta. Ogni hop legge al massimo 100 archi con una query
aggiuntiva di un arco per rilevare il troncamento; resta il limite di output di 64000 caratteri.
I vicini non includono abstract/summary/testi completi dei nodi: la query SQL legge
direttamente una proiezione compatta, evitando di caricare 100 abstract per una navigazione.
Il limite di 100 riguarda i vicini, non un conteggio totale certificato dei collegamenti.

Il quinto round (o il round successivo al sesto tentativo tool) forza `tool_choice=none`
per riservare una risposta finale nei budget esistenti. Se i risultati non bastano,
il modello deve dichiararlo, senza inventare ranking. Questo non amplia timeout o token.
I fallimenti applicativi noti espongono codici controllati anche nei gruppi MCP:
`agent_tool_budget`, `agent_token_budget`, `agent_round_budget`, `gateway_output_limit`,
`gateway_incomplete_answer`, `sql_read_failed`, `genie_query_failed`, `tool_capacity`,
`tool_response_too_large`. Il codice non identifica da solo la causa interna del provider:
non si fanno retry automatici degli errori SQL/Genie/Gateway o fallback di autenticazione.

Il Gateway comune non registra/pubblica i nostri tool: le descrizioni vengono inviate
nelle singole richieste. Chi amministra Gateway/workspace puo' comunque avere accesso
a risorse e log: questa architettura non promette isolamento dagli amministratori,
ne' invisibilita' del nome delle risorse in tutte le schermate.

#### Primo deployment in dev

Prerequisiti aggiuntivi:

1. Apps e Managed MCP Servers disponibili/abilitati nel workspace.
2. CLI Databricks recente che riconosca `apps.config` e il comando Bundle per avviare App:
   non ignorare warning su campi sconosciuti.
3. Identita' di setup autorizzata a creare viste/grant UC, gestire Genie/ACL, creare ruoli
   OAuth e concedere SELECT in Lakebase.
4. **Utente proprietario autorizzato sul Gateway del corso** per chat ed embedding:
   EXECUTE sui model services `course_ops.gateway.chat` e `course_ops.gateway.embed`,
   USE CATALOG/USE SCHEMA sui contenitori, anche tramite gruppo studenti.
   Il job non cambia ACL/budget del Gateway condiviso e non usa PAT personali salvati.
5. Accesso API Lakebase/generazione credenziali e rete dalla App all'host PostgreSQL.
   I grant UC non sostituiscono le autorizzazioni Lakebase.
6. Per l'orchestratore: proprietario/run-as autorizzato a eseguire entrambi i job figli,
   leggere i relativi output e gestire stop/start/deploy della App. In dev usare lo stesso
   proprietario umano del setup; non eseguire una riconfigurazione manuale in parallelo.
7. User Authorization disponibile per gli scope `sql`, `genie` e `ai-gateway`. Alla prima apertura
   accettare il consenso con l'utente proprietario, che deve gia' poter usare il Warehouse
   e leggere le viste curate. Non serve concedere CAN_USE del Warehouse alla App.

```powershell
databricks bundle validate -t dev
databricks bundle deploy -t dev

# Percorso completo: nessun ID da recuperare manualmente.
# Prima verifica accesso Gateway dell'utente e consenso agli scope App.
databricks bundle run -t dev refresh_assistant --params "topic=graph neural networks,num_papers=10"
```

Il job [refresh_assistant](resources/assistant_orchestrator.job.yml) riusa i due job
esistenti, senza copiare task della pipeline:

```text
pipeline (Run Job: ingest_papers)
  -> resolve_collection (output della specifica esecuzione figlia)
  -> stop_assistant (attende compute STOPPED)
  -> configure_assistant (Run Job: setup_assistant, con run_id + graph_build_id)
  -> start_assistant (attende RUNNING e verifica /api/collection autenticato)
```

Parametri: `env`, `topic`, `num_papers`, `ranking_criteria`, `criteria_date`, `categories`,
con gli stessi default della pipeline. Il job ha una sola esecuzione concorrente e coda
abilitata. Non programma esecuzioni automatiche: si avvia dalla Jobs UI o dal comando sopra.
La App viene sempre **creata dal Bundle deploy**, non dal job.

Il Run Job restituisce l'ID Databricks dell'esecuzione figlia. Il task di raccordo legge
il suo `build_graph`, verifica job/stato/ambiente e valida il piccolo JSON versionato
restituito dal notebook con `dbutils.notebook.exit`. Pubblica quindi gli ID della raccolta
come task values del padre. ID Databricks e `run_id` applicativo sono distinti; nessuna
query all'ultima run, parsing dei log o propagazione implicita di task values fra job.
Serve ridistribuire anche il notebook `build_graph` aggiornato.

La vecchia raccolta resta disponibile durante ingestion. Solo dopo pipeline e raccordo
riusciti la App viene fermata. Al primo avvio, senza deployment precedente, il job
distribuisce uno snapshot dei sorgenti Bundle con il comando di `app.yaml` e le variabili
del target. Negli avvii successivi **riavvia il deployment esistente**, senza sostituirne
il codice. Un aggiornamento del codice App si distribuisce esplicitamente con
`databricks bundle run -t dev bibliographic_assistant`.

Tutti i task dipendono dal successo del precedente:

- Fallimento pipeline/raccordo: niente stop o setup, App precedente invariata.
- Fallimento stop: niente setup; non si assume che il servizio sia fermo.
- Fallimento setup: niente riavvio; App ferma, correggere e usare **Repair** sul task
  `configure_assistant` con i dipendenti, mantenendo pipeline/raccordo riusciti.
- Fallimento avvio/verifica HTTP: correggere e riparare `start_assistant`, senza ingestion.
  Potrebbe esserci gia' un processo attivo: il Repair verifica anche la raccolta esposta,
  non crea un secondo deployment se quello precedente esiste.
- Nuova esecuzione completa: nuova ingestion e nuova raccolta. Non usarla al posto di un
  Repair se si vuole conservare il risultato gia' prodotto. Niente retry automatici dei
  task Run Job che possano creare nuove ingestion.

La verifica finale usa OAuth del proprietario attraverso il proxy Apps, senza falsificare
header identita', e confronta `run_id`/`graph_build_id` della risposta HTTP. Non basta compute
ACTIVE e non si verifica solo che l'URL esista. Non esegue ancora una domanda al modello:
permessi Gateway e qualita' delle risposte vanno collaudati separatamente. Con User
Authorization il proxy deve inoltrare anche un token con scope SQL/Genie/Gateway: se il controllo
da notebook non dispone di tale autorizzazione, il task fallisce esplicitamente.
Aprire l'App come proprietario, completare il consenso e verificare il flusso OAuth
prima di considerare collaudato l'avvio automatizzato; non falsificare header di identita'.
La coda serializza questo orchestratore, non eventuali setup/deploy manuali: evitare
operazioni concorrenti sulla stessa App.

Per una raccolta gia' pronta, rimane disponibile il percorso manuale senza nuova ingestion:

```powershell
# Se l'App e' gia' attiva, fermala prima di riconfigurarla.
databricks bundle run -t dev setup_assistant --params "run_id=<RUN_ID>,graph_build_id=<GRAPH_BUILD_ID>"

# Avvia/deploya il servizio persistente dopo il setup.
databricks bundle run -t dev bibliographic_assistant
```

Il deploy iniziale crea la risorsa App e la sua identita'; non e' necessario avviare
l'assistente prima di avere configurato i grant. Aprire poi l'URL della App come
proprietario e accettare il consenso SQL/Genie/Gateway.
Il processo non esegue SQL prima di una richiesta utente. Senza setup la prima richiesta
fallisce esplicitamente: nessuna raccolta di default.

Per cambiare raccolta: fermare la App, rieseguire `setup_assistant` con nuovi ID,
poi ridistribuire/riavviare la App. Non e' un'operazione multi-risorsa atomica: se il setup
fallisce, correggere il prerequisito e ripeterlo **prima** di riavviare. Il manifest
viene scritto solo dopo configurazione e grant; nessun fallimento viene presentato come successo.
Per ora il proprietario e' l'identita' che esegue il setup: usare il proprio utente in dev.
L'uso con deployment service principal acc/prd richiede di definire esplicitamente un
proprietario umano/allowlist e autorizzazioni appropriate; non e' un servizio multiutente.

La App usa Python **3.12** tramite `uv` e il lock del repository, non il Python 3.11
di default delle Apps basate su pip. L'extra opzionale `app` dichiara MCP/Starlette/Uvicorn
alle versioni gia' presenti nel lock; nessuna versione esistente aggiornata.
Non aggiungere `requirements.txt` alla root: avrebbe precedenza su `pyproject.toml`
e riporterebbe il deployment al percorso pip/Python 3.11.

#### Spegnimento programmato della App

Il job [shutdown_assistant](resources/assistant_shutdown.job.yml) ha un solo task:
riusa `assistant_lifecycle.py` con `action=stop` sulla App del target. Viene eseguito
**ogni giorno alle 13:00 e alle 20:00, fuso `Europe/Rome`**, anche nel weekend;
l'ora legale/solare e' gestita dal fuso, non da offset UTC fissi.
Il cron Quartz e' `0 0 13,20 * * ?`, con `pause_status: UNPAUSED` esplicito per
attivare la schedule anche nel target dev, che altrimenti sospende le schedule.
Si attiva al prossimo deploy del Bundle; non e' necessario avviare il job manualmente.

Lo stop e' ripetibile: se la App e' gia' ferma, termina senza riavviarla.
Ferma il compute App, non cancella risorse/dati e non spegne Warehouse o Lakebase.
Puo' interrompere chat in corso; l'orario e' quello del trigger, non una garanzia
che il compute sia gia' fermo al secondo esatto (startup job e stop richiedono tempo).
Il run-as del job deve poter gestire la App; errori di stop/deployment pendente fanno
fallire il task, non vengono nascosti.

Non e' un blocco all'uso fuori orario: Start manuale o `refresh_assistant` possono
riaccenderla dopo lo spegnimento, anche se il refresh termina dopo le 13/20.
Evitare refresh/deploy in prossimita' degli orari di stop: i due job sono indipendenti.
Per riavviare la stessa raccolta senza ingestion:

```powershell
databricks apps start autographrs-rtambone-dev
```

#### Setup: crash durante import psycopg

Un SIGABRT con stack in `psycopg/pq/__init__.py` durante l'import di `psycopg_binary`
precede le operazioni del setup, quindi non e' una risposta di autorizzazione SQL.
Il notebook di setup seleziona `PSYCOPG_IMPL=python` prima degli import applicativi,
evitando l'estensione binaria in quel processo e usando la `libpq` di sistema.
Stampa implementazione/versione libpq prima del provisioning; nessuna versione di
pacchetto o runtime modificata, nessun cambio al percorso della pipeline o della App.
Questa mitigazione richiede una libpq di sistema disponibile: se manca, l'import
fallisce esplicitamente e occorre valutare il runtime, non installare pacchetti a caso.
Ridistribuire il Bundle e riparare il solo setup con una nuova sessione Python.
Il comportamento nel runtime remoto va verificato; i test locali non riproducono il SIGABRT.

#### Dove vive l'agente e differenza dalla scheda Agents

L'orchestratore bibliografico e' un **agente custom Python nella Databricks App**:
il suo loop tool-calling chiama il modello chat del Gateway del corso. Non e' un
modello nuovo, un endpoint Model Serving o una registrazione automatica nella scheda
Agents/Agent Bricks. I template agent della UI sono un altro percorso di creazione
e possono aggiungere Agent Server, MLflow, tracing e risorse gestite che qui non
abbiamo introdotto.

Il **Genie Agent** creato dal setup e' una risorsa distinta, visibile nella UI Genie
secondo le ACL del workspace: interpreta domande analitiche sulle viste curate.
L'agente custom lo richiama tramite `ask_collection_analytics`/MCP; non ospita il
nostro codice, gli altri quattro tool o il nostro loop Gateway. I modelli interni
di Genie sono gestiti da Databricks, non configurati dal nostro endpoint chat.

#### MCP, limiti e verifica

Endpoint custom: **`<APP_URL>/mcp`**, protocollo Streamable HTTP. Client esterni devono
autenticarsi come proprietario via OAuth Databricks e avere accesso alla App;
non basta conoscere l'URL. L'assistente chiama direttamente i servizi Python condivisi,
non compie un giro HTTP su se stesso. Per Genie usa invece
`https://<workspace>/api/2.0/mcp/genie/<space_id>`, con l'identita' della App.
Nessuna registrazione dei tool nel catalogo MCP condiviso del Gateway.

Limiti imposti dal codice:

- Solo messaggi client `user`/`assistant`, massimo 12 messaggi e 16000 caratteri totali;
  niente prompt `system` o risultati `tool` forniti dal client.
- Massimo 5 round del modello, 6 chiamate tool, risposte modello di massimo 1600 token.
  Ogni round reinvia l'intera conversazione, quindi il budget e' per richiesta:
  `prompt_tokens` di un singolo round massimo 48000 (`agent_context_budget`) e tetto
  complessivo di sicurezza 160000 token (`agent_token_budget`). I risultati tool inviati
  al modello sono limitati a 128000 caratteri cumulativi: oltre, il risultato non entra nel
  contesto, il modello riceve `evidence_budget_exhausted` e il round successivo e' solo risposta.
- Budget di orchestrazione 120s verificato tra operazioni; chiamate Gateway con timeout
  massimo 30s e senza retry, embedding 20s, Genie MCP 40s complessivi, SQL con polling
  limitato e cancellazione esplicita, statement PostgreSQL 20s. Il budget non e' una
  terminazione forzata di operazioni remote gia' in corso o un tetto monetario garantito.
- Due chat concorrenti e due esecuzioni tool per processo; richieste oltre capacita' rifiutate.
- Top-k semantico massimo 20; bibliografia paginata a massimo 30; vicini default/massimo 100,
  profondita' 2 e batch archi limitati, con `truncated` esplicito.
- Corpi HTTP/MCP massimo 64000 byte, output di ciascun tool massimo 64000 caratteri:
  output piu' grandi non vengono troncati silenziosamente; in chat diventano
  `tool_response_too_large`, feedback correggibile dal modello con un limite minore.
- Niente CORS permissivo, controllo Origin, CSP e rendering testuale; errori backend
  esposti con tipo/riferimento, senza restituire stack trace o credenziali.

Validazione locale: **240 test superati**, inclusi HTTP/MCP, loop tool calling, errori
Genie, isolamento della raccolta, grant, snapshot vuoti, handoff fra job e lifecycle App;
isolamento OAuth utente per richiesta e separazione dalle credenziali App;
Ruff e build wheel superati.
Nessuna versione dei 158 pacchetti nel lock modificata. YAML verificati contro lo schema
pubblico del CLI Databricks; questa verifica strutturale non sostituisce
`databricks bundle validate` autenticato o i test reali nel workspace.
Rimane un warning di deprecazione del TestClient Starlette con HTTPX, senza aggiornare
le versioni dell'ambiente del corso.

Prima di considerare privato il deployment, verificare da un'identita' non autorizzata:
App e `/mcp` negati, Genie negato, letture dirette UC/Lakebase negate. Controllare anche
grant ereditati, gruppi e visibilita' delle cartelle workspace. I test locali simulano
gli header del proxy e verificano l'allowlist, non dimostrano le ACL reali del workspace.

Costi: compute App mentre in esecuzione, Gateway, Warehouse e attivita' Lakebase.
Partire da dev e fermare la App quando non serve. `bundle destroy` non elimina lo spazio
Genie creato dal job, le viste/manifest o i ruoli/grant SQL: la dismissione e' esplicita,
senza cancellazioni automatiche di dati o altre risorse del corso.

#### MLflow rimandato

**Nessuna integrazione MLflow aggiunta**, su richiesta: niente autologging, tracking,
tracing, registrazione modello o persistenza delle conversazioni. Lo storico chat
rimane in memoria nel browser; il server restituisce le evidenze alla singola richiesta.
Restano possibili i log di piattaforma/Gateway configurati dagli amministratori.
MLflow, valutazione retrieval/risposte e metriche di qualita' saranno un incremento successivo.

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

In the deployed **job-autograph-pipeline** job, choose **Run now with different parameters**:
set `topic` to your search, `num_papers` to `3`, `ranking_criteria` to `relevance`,
and leave `criteria_date`, `categories` and `run_id` empty for the first trial.
Keep `env=dev`. The final notebook cell verifies that published paper and enrichment
counts match; check `pipeline_runs.warnings` for external API limitations.

### Topic: concetti separati da virgola

Nel parametro `topic` separare i concetti con **`,`**, per esempio:

```text
neuroscience, peririnal cortex
```

Il provider arXiv traduce questo input in:

```text
(all:"neuroscience" AND all:"peririnal cortex")
```

Ogni concetto e' una frase, senza bisogno di virgolette; devono comparire **tutti**.
Non sono ricerche indipendenti e `num_papers` limita il totale, non il numero per concetto.
Piu' concetti rendono la ricerca piu' restrittiva, quindi possono dare pochi risultati.
Le virgole iniziali/finali, i concetti vuoti e le virgolette nelle liste vengono rifiutati.
Nelle liste gli operatori sono testo letterale, non sintassi del provider.

Senza virgole il comportamento precedente rimane invariato: e' possibile usare una query
arXiv esplicita, per esempio `all:neuroscience OR all:"peririnal cortex"`.
Il testo originale ripulito dagli spazi esterni rimane nella richiesta salvata; la traduzione
e' responsabilita' del provider di ricerca, non dei servizi bibliografici Scholar/OpenAlex.

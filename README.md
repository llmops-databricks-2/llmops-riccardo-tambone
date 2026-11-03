# llmops-riccardo-tambone

Your repository for the LLMOps course on Databricks. Everything you need in
the course workspace was provisioned for you before you got this repo, and every id is already in
`project_config.yml`: you configure nothing.

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
`ai_query` refuses the model service.

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

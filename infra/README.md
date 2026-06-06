# Infrastructure (Azure)

Deploys the scoring API to **Azure Container Apps** via Bicep: an Azure Container Registry,
a Log Analytics workspace, a Container Apps managed environment, and the autoscaling
`scoring-api` app (HTTP-based scale 1→3, `/health` liveness + readiness probes).

```bash
RG=rg-fraud-prod
az group create -n $RG -l australiaeast

# 1. provision ACR first (so we have somewhere to push the image)
az deployment group create -g $RG -f infra/bicep/main.bicep \
  -p infra/bicep/parameters.prod.json -p apiContainerImage=placeholder

# 2. build & push the API image
ACR=$(az acr list -g $RG --query "[0].name" -o tsv)
az acr build -r $ACR -t fraud-api:latest -f docker/api.Dockerfile .

# 3. deploy for real, pointing at the pushed image
az deployment group create -g $RG -f infra/bicep/main.bicep \
  -p infra/bicep/parameters.prod.json \
  -p apiContainerImage=$ACR.azurecr.io/fraud-api:latest

# the API URL is in the deployment outputs
az deployment group show -g $RG -n main --query properties.outputs.apiUrl.value -o tsv
```

> The model artifact is baked into the image (or pulled from the MLflow registry at startup
> via `FRAUD_MLFLOW_TRACKING_URI` + `FRAUD_MODEL_STAGE`). For a managed MLflow + Postgres,
> add Azure Database for PostgreSQL and an MLflow Container App alongside this template.

targetScope = 'resourceGroup'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Short project identifier used in resource naming.')
param projectName string = 'fraud'

@description('Environment suffix used in resource naming.')
param environmentName string = 'prod'

@description('Short region code used in resource naming.')
param regionCode string = 'eaus2'

@description('ACR SKU.')
@allowed(['Basic', 'Standard', 'Premium'])
param acrSku string = 'Basic'

@description('Scoring API container image (in the ACR), e.g. <acr>.azurecr.io/fraud-api:latest.')
param apiContainerImage string

@description('Requested CPU cores for the scoring Container App.')
param apiCpu int = 1

@description('Requested memory for the scoring Container App.')
param apiMemory string = '2Gi'

@description('MLflow registry stage the API serves.')
param modelStage string = 'Production'

var acrName = 'acr${projectName}${environmentName}${regionCode}'
var logName = 'log-${projectName}-${environmentName}-${regionCode}'
var envName = 'cae-${projectName}-${environmentName}-${regionCode}'
var apiAppName = 'ca-${projectName}-api-${environmentName}-${regionCode}'

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: { name: acrSku }
  properties: { adminUserEnabled: true }
}

resource logws 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource caenv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logws.properties.customerId
        sharedKey: logws.listKeys().primarySharedKey
      }
    }
  }
}

resource apiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: apiAppName
  location: location
  properties: {
    managedEnvironmentId: caenv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'scoring-api'
          image: apiContainerImage
          env: [
            { name: 'FRAUD_MODEL_STAGE', value: modelStage }
            { name: 'FRAUD_API_PORT', value: '8000' }
          ]
          resources: {
            cpu: apiCpu
            memory: apiMemory
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 30
              periodSeconds: 15
            }
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 15
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaling'
            http: { metadata: { concurrentRequests: '50' } }
          }
        ]
      }
    }
  }
}

output acrLoginServer string = acr.properties.loginServer
output apiUrl string = 'https://${apiApp.properties.configuration.ingress.fqdn}'

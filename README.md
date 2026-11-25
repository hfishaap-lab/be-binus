# AGRAPATHA — Vision AI Backend (Railway Deployment)

MobileNetV2 TFLite wildfire & smoke detection microservice for ESP32-CAM and React Web Dashboard.

## Endpoints

| Method | Endpoint | Description | Payload |
|---|---|---|---|
| `GET` | `/` | Root service metadata | None |
| `GET` | `/health` | Health & model check | None |
| `POST` | `/api/v1/predict` | Classification inference | Multipart form `file` or raw binary JPEG |
| `POST` | `/predict/fire` | ESP32-CAM dedicated alias | Multipart form `file` or raw binary JPEG |

## Classes
1. `Fire` (`FIRE`)
2. `Smoke` (`SMOKE`)
3. `Non-Fire` (`SAFE`)

## Deployment to Railway
1. Push to `https://github.com/hfishaap-lab/be-binus`
2. In Railway, click **Deploy**. Railway will detect `Dockerfile` automatically and build using Python 3.11.

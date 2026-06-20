# CloudFlow CRM API Authentication Guide

Welcome to the CloudFlow CRM API integration documentation. All API endpoints require authentication using a Bearer token or OAuth2 flow.

## 1. Bearer Token Authentication
To authenticate requests, generate an API Key from the CloudFlow CRM Dashboard (Settings > Developer > API Keys) and pass it in the `Authorization` header of all HTTP requests:

```http
Authorization: Bearer <YOUR_API_KEY>
```

### Response Codes:
* `200 OK`: Request succeeded.
* `401 Unauthorized`: Invalid or expired API key.
* `403 Forbidden`: Insufficient permissions (e.g., trying to modify billing endpoints with a read-only token).

## 2. Token Troubleshooting
If you encounter a `401 Unauthorized` response:
1. Validate that the token does not contain extra spaces or prefix quotes.
2. Confirm the token is not expired (tokens generated via the UI have a default lifespan of 180 days unless configured otherwise).
3. Ensure the base URL is correct: `https://api.cloudflowcrm.com/v2/`.

## 3. Webhooks Security
All webhooks emitted by CloudFlow CRM are signed with a HMAC-SHA256 signature passed in the `X-CloudFlow-Signature` header. Compute the signature of the payload using your Webhook Secret Key to verify authenticity.

# CloudFlow CRM Data Security & Compliance Policy

CloudFlow CRM is dedicated to safeguarding customer data. This document outlines our data protection, storage, and regulatory compliance standards.

## 1. Data Encryption
* **Data at Rest**: All customer databases, configurations, and RAG search indices are encrypted at rest using AES-256 (Advanced Encryption Standard with a 256-bit key). Keys are managed automatically via AWS KMS with automatic annual rotation.
* **Data in Transit**: All connections to CloudFlow CRM services must use Transport Layer Security (TLS 1.3 or TLS 1.2 minimum). Connections over unencrypted HTTP (port 80) are automatically redirected to HTTPS (port 443).

## 2. Regulatory Compliance
* **GDPR (General Data Protection Regulation)**: Users can request complete deletion of their account and personal data (Right to be Forgotten) by opening an admin ticket. Data deletion is processed and finalized across all databases within 30 business days.
* **SOC 2 Type II**: CloudFlow CRM is audited annually. The compliance report is available upon request for Enterprise customers under non-disclosure agreements (NDA).

## 3. Account Access Recovery
For security reasons, password resets can only be initiated via the registered email address. Support agents cannot manually change or set passwords for user accounts. If a user is locked out, they must wait 15 minutes or trigger the automated password reset form.

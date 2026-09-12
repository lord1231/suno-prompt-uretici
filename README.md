---
title: Suno Prompt Uretici
emoji: müzik
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "6.26.0"
app_file: app.py
pinned: false
---

# Suno Prompt Uretici

Bir sarki yukle, teknik analiz (BPM/Key/Akor/Waveform), AI destekli derin muzikal analiz, tum stemler ve Suno icin hazir prompt otomatik uretilsin.

## Bu repo Hugging Face Space'ine otomatik senkronize olur

GitHub'da main dalina yapilan her push, .github/workflows/deploy.yml uzerinden otomatik olarak Hugging Face Space'ine gonderilir.

Space'in kendi Secrets'inda (Settings > Repository secrets) sunlar tanimli olmali: FADR_API_KEY, GEMINI_API_KEY, APP_PASSWORD

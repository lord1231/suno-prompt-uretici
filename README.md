[README.md](https://github.com/user-attachments/files/32140277/README.md)
title: Suno Prompt Uretici
emoji: 🎧
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "6.26.0"
app_file: app.py
pinned: false
---

# Suno Prompt Üretici

Bir şarkı yükle, teknik analiz (BPM/Key/Akor/Waveform), AI destekli derin
müzikal analiz, tüm stem'ler ve Suno için hazır prompt otomatik üretilsin.

## Bu repo Hugging Face Space'ine otomatik senkronize olur

GitHub'da `main` dalına yapılan her push, `.github/workflows/deploy.yml`
üzerinden otomatik olarak Hugging Face Space'ine gönderilir.

Space'in kendi Secrets'ında (Settings > Repository secrets) şunlar
tanımlı olmalı:
- `FADR_API_KEY`
- `GEMINI_API_KEY`
- `APP_PASSWORD`

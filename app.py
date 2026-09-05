# ============================================================
# 🎧 FADR + GEMINI MUSIC ANALYZER — HUGGING FACE SPACES SÜRÜMÜ
# Kalıcı web uygulaması, şifre korumalı.
# Bu dosyayı Space'e app.py olarak yükle; requirements.txt ve
# packages.txt dosyaları ayrıca gerekiyor (aşağıda ayrı verildi).
# ============================================================

import os
import inspect

SOUNDFONT_PATH = "/tmp/soundfont.sf3"

def ensure_soundfont():
    """Ses fontunu SADECE gerçekten ihtiyaç duyulduğunda indirir.
    Açılışta indirmek, HF Spaces'in sağlık kontrolü zaman aşımına
    uğramasına ve sürekli yeniden başlamaya (crash loop) yol açıyordu."""
    if not os.path.exists(SOUNDFONT_PATH):
        print("🎹 Ses fontu indiriliyor...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3",
            SOUNDFONT_PATH,
        )
        print("✅ Ses fontu hazır.")

import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import StringIO
import time
import uuid
from google import genai
from google.genai import types
from midi2audio import FluidSynth
from pydub import AudioSegment
import gradio as gr

# ------------------------------------------------------------
# API ANAHTARLARI — Space Secrets'tan okunur (Settings > Secrets)
# ------------------------------------------------------------

FADR_API_KEY = os.environ.get("FADR_API_KEY", "")
headers = {"Authorization": f"Bearer {FADR_API_KEY}"}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

client = genai.Client(api_key=GEMINI_API_KEY)

# ------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ------------------------------------------------------------

def download_asset_file(asset_id, quality="download"):
    r = requests.get(f"https://api.fadr.com/assets/download/{asset_id}/{quality}", headers=headers, timeout=30)
    r.raise_for_status()
    file_url = r.json()["url"]
    fr = requests.get(file_url, timeout=60)
    fr.raise_for_status()
    return fr

import re

TURKISH_CHAR_MAP = str.maketrans({
    "ş": "s", "Ş": "S", "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U",
})

def clean_filename_part(text):
    text = text.translate(TURKISH_CHAR_MAP)
    text = re.sub(r'[\\/:*?"<>|]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_artist_name(song_name):
    base = os.path.splitext(song_name)[0]
    if " - " in base:
        artist = base.split(" - ")[0]
    else:
        artist = base
    return clean_filename_part(artist)

def simplify_chord(chord):
    replacements = {
        "D:min": "Dm", "C:min": "Cm", "F:maj": "F", "G:min": "Gm",
        "D#:maj": "Eb", "A#:maj": "Bb", "C:maj": "C", "G:maj": "G",
        "A:maj": "A", "A:min": "Am", "E:min": "Em", "B:min": "Bm",
    }
    return replacements.get(str(chord), str(chord))

def generate_with_retry(audio_file, user_prompt, system_instruction, max_retries=5, initial_wait=10):
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[types.Content(role="user", parts=[
                    types.Part.from_uri(file_uri=audio_file.uri, mime_type=audio_file.mime_type),
                    types.Part.from_text(text=user_prompt),
                ])],
                config=types.GenerateContentConfig(system_instruction=system_instruction),
            )
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(initial_wait * attempt)

SYSTEM_INSTRUCTION = """
Sen deneyimli bir müzik prodüktörü, tonmeister ve müzikoloğusun. Sana bir
şarkının sesi ve o şarkı hakkındaki teknik (Fadr) verileri veriliyor.

Görevin:
1) Şarkıyı gerçekten dinleyip aşağıdaki TÜM kategorilerde detaylı,
   teknik ve isabetli bir analiz yapmak.
2) Bu analize dayanarak Suno AI için 1000 karakteri KESİNLİKLE
   AŞMAYAN bir prompt üretmek. Suno promptu; tür, ruh hali, tempo,
   enstrümantasyon, VOKAL TARZI/KARAKTERİ ve prodüksiyon stilini
   MUTLAKA içermeli.

Cevabını TAM OLARAK şu formatta ver:

=== MÜZİK ANALİZİ ===
Tür: ...
Alt tür: ...
Dönem: ...
Vokal: (cinsiyet, ton aralığı, tını, ifade biçimi, teknik özellikler)
Enstrümanlar: (tespit edilen tüm enstrümanlar, öne çıkanlar)
Ritim: (groove karakteri, senkop, vurgu düzeni)
Armoni: (modal/tonal yapı, gerilim-çözülme karakteri)
Enerji Seviyesi: (düşük/orta/yüksek, şarkı boyunca değişim var mı)
Dinamik Aralık: (sıkışık/geniş, ne kadar "nefes alan" bir mikslemesi var)
Dans Edilebilirlik: (1-10 arası tahmini puan ve kısa gerekçe)
Prodüksiyon Teknikleri: (reverb/delay kullanımı, stereo genişlik, katmanlama, analog/dijital karakter)
Mix Karakteri: (vokal öne mi çıkıyor, bas ağırlıklı mı, parlak mı karanlık mı)
Benzer Tarzlar: (isim vermeden, hangi alt türlere/akımlara yakın)

=== SUNO PROMPT ===
(1000 karakterden kısa, tek paragraf, vokal tarzını da içeren prompt)
"""

# ------------------------------------------------------------
# ANA ANALİZ FONKSİYONU
# ------------------------------------------------------------

def clear_outputs():
    waiting = "⏳ Analiz başlıyor, lütfen bekle... (1-3 dakika sürebilir)"
    empty_df = pd.DataFrame(columns=["start", "end", "chord", "chord_basit"])
    return waiting, "", "", "", "", empty_df, None, "", None, None, None, None, None, None, None, None

def analyze_song_pipeline(local_file_path, progress=gr.Progress()):
    empty_df = pd.DataFrame(columns=["start", "end", "chord", "chord_basit"])
    if local_file_path is None:
        return ("⚠️ Lütfen önce bir şarkı dosyası yükle.", "", "", "", "", empty_df, None, "",
                None, None, None, None, None, None, None, None)
    try:
        return _run_pipeline(local_file_path, progress)
    except Exception as e:
        err = f"❌ Analiz sırasında hata oluştu:\n\n{type(e).__name__}: {e}"
        return (err, "", "", "", "", empty_df, None, "",
                None, None, None, None, None, None, None, None)

def _run_pipeline(local_file_path, progress):
    song_name = os.path.basename(local_file_path)
    print(f"✅ İşlenen dosya: {song_name}")
    ext = os.path.splitext(song_name)[1].lower().replace(".", "")

    progress(0.05, desc="Fadr'a yükleniyor...")
    r = requests.post("https://api.fadr.com/assets/upload2", json={"name": song_name, "extension": ext}, headers=headers)
    r.raise_for_status()
    upload_info = r.json()

    with open(local_file_path, "rb") as f:
        file_data = f.read()
    mime_types = {"mp3": "audio/mp3", "wav": "audio/wav", "m4a": "audio/mp4", "aac": "audio/aac", "aif": "audio/aiff", "flac": "audio/flac"}
    mime = mime_types.get(ext, "application/octet-stream")
    requests.put(upload_info["url"], data=file_data, headers={"Content-Type": mime}).raise_for_status()

    r = requests.post("https://api.fadr.com/assets", json={"name": song_name, "extension": ext, "group": song_name + "-group", "s3Path": upload_info["s3Path"]}, headers=headers)
    r.raise_for_status()
    asset_id = r.json()["asset"]["_id"]
    print(f"✅ Fadr Asset ID: {asset_id}")

    progress(0.15, desc="Fadr AI analiz ediyor (stem/akor/tempo)...")
    r = requests.post("https://api.fadr.com/assets/analyze/stem", json={"_id": asset_id}, headers=headers)
    r.raise_for_status()
    task_id = r.json()["task"]["_id"]

    while True:
        r = requests.post("https://api.fadr.com/tasks/query", json={"_ids": [task_id]}, headers=headers)
        r.raise_for_status()
        if r.json()["tasks"][0].get("status", {}).get("complete"):
            break
        time.sleep(5)

    progress(0.4, desc="Sonuçlar toplanıyor...")
    r = requests.get(f"https://api.fadr.com/assets/{asset_id}", headers=headers, timeout=30)
    r.raise_for_status()
    final_asset = r.json()["asset"]
    tempo = final_asset["metaData"].get("tempo")
    key = final_asset["metaData"].get("key")
    artist_name = extract_artist_name(final_asset.get("name", song_name))

    stem_assets = []
    for stem_id in final_asset.get("stems", []):
        r = requests.get(f"https://api.fadr.com/assets/{stem_id}", headers=headers, timeout=30)
        r.raise_for_status()
        stem_assets.append(r.json()["asset"])
    stem_types = [s.get("metaData", {}).get("stemType") for s in stem_assets if s.get("metaData", {}).get("stemType")]

    midi_assets = []
    for midi_id in final_asset.get("midi", []):
        r = requests.get(f"https://api.fadr.com/assets/{midi_id}", headers=headers, timeout=30)
        r.raise_for_status()
        midi_assets.append(r.json()["asset"])

    chord_csv_asset = next((m for m in midi_assets if m.get("assetType") == "chord-csv"), None)
    chord_progression = ""
    df_chords_display = pd.DataFrame(columns=["chord", "start", "end"])
    if chord_csv_asset:
        chord_csv = download_asset_file(chord_csv_asset["_id"], "download").text
        df_chords = pd.read_csv(StringIO(chord_csv))
        df_chords["chord_basit"] = df_chords["chord"].apply(simplify_chord)
        df_chords["start"] = df_chords["start"].round(2)
        df_chords["end"] = df_chords["end"].round(2)
        df_chords_display = df_chords[["start", "end", "chord", "chord_basit"]]
        chords = []
        for chord in df_chords["chord"].tolist():
            simple = simplify_chord(chord)
            if not chords or chords[-1] != simple:
                chords.append(simple)
        chord_progression = " → ".join(chords[:40])

    progress(0.55, desc="Melodi mp3'leri hazırlanıyor...")
    ensure_soundfont()
    melody_labels = {"vocals": "Vokal", "other": "Enstruman"}
    melody_paths = {"vocals": None, "other": None}
    for m in midi_assets:
        midi_type = m.get("metaData", {}).get("midiType")
        if midi_type in melody_labels:
            try:
                midi_bytes = download_asset_file(m["_id"], "download").content
                file_base = f"{artist_name} - {melody_labels[midi_type]} (Sentez)"
                midi_path = f"/tmp/{file_base}.mid"
                wav_path = f"/tmp/{file_base}.wav"
                mp3_path = f"/tmp/{file_base}.mp3"
                with open(midi_path, "wb") as f:
                    f.write(midi_bytes)
                FluidSynth(sound_font=SOUNDFONT_PATH).midi_to_audio(midi_path, wav_path)
                AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3")
                melody_paths[midi_type] = mp3_path
            except Exception:
                pass

    # ---- TÜM STEM'LERİ İNDİR (vokal, bas, davul, diğer, enstrümantal) ----
    progress(0.6, desc="Tüm stem'ler indiriliyor...")
    stem_label_map = {
        "vocals": "Vokal", "bass": "Bas", "drums": "Davul",
        "other": "Diger", "instrumental": "Enstrumantal",
    }
    stem_paths = {"vocals": None, "bass": None, "drums": None, "other": None, "instrumental": None}
    for s in stem_assets:
        s_type = s.get("metaData", {}).get("stemType")
        if s_type in stem_paths:
            try:
                audio_bytes = download_asset_file(s["_id"], "hqPreview").content
                label = stem_label_map.get(s_type, s_type)
                path = f"/tmp/{artist_name} - {label} (Stem).mp3"
                with open(path, "wb") as f:
                    f.write(audio_bytes)
                stem_paths[s_type] = path
            except Exception:
                pass
    real_vocal_path = stem_paths["vocals"]

    fadr_report = f"""
FADR TEKNİK ANALİZİ
Şarkı: {song_name}
Tempo: {tempo} BPM
Key: {key}
Stem'ler: {", ".join(stem_types)}
Akor progresyonu (özet): {chord_progression}
"""

    progress(0.7, desc="Gemini şarkıyı dinliyor...")
    audio_content = download_asset_file(asset_id, "hqPreview").content
    local_audio_path = f"/tmp/{artist_name} - Orijinal.mp3"
    with open(local_audio_path, "wb") as f:
        f.write(audio_content)

    # ---- WAVEFORM GÖRSELİ + SES İSTATİSTİKLERİ ----
    waveform_path = None
    audio_stats_md = ""
    try:
        seg = AudioSegment.from_file(local_audio_path)
        avg_dbfs = seg.dBFS
        peak_dbfs = seg.max_dBFS
        dynamic_range = peak_dbfs - avg_dbfs
        duration_sec = len(seg) / 1000

        samples = np.array(seg.get_array_of_samples())
        if seg.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        step = max(1, len(samples) // 3000)
        samples_ds = samples[::step]

        plt.figure(figsize=(10, 2.2))
        plt.plot(samples_ds, linewidth=0.6, color="#7c3aed")
        plt.axis("off")
        plt.tight_layout(pad=0)
        waveform_path = f"/tmp/{artist_name} - waveform.png"
        plt.savefig(waveform_path, dpi=110, bbox_inches="tight", transparent=True)
        plt.close()

        audio_stats_md = (
            "### 📈 Teknik Ses İstatistikleri\n"
            f"| Ölçüm | Değer |\n|---|---|\n"
            f"| Süre | {duration_sec:.1f} saniye |\n"
            f"| Ortalama Ses Seviyesi | {avg_dbfs:.1f} dBFS |\n"
            f"| Tepe (Peak) Seviye | {peak_dbfs:.1f} dBFS |\n"
            f"| Dinamik Aralık | {dynamic_range:.1f} dB |\n"
            f"| Kanal Sayısı | {seg.channels} ({'Stereo' if seg.channels == 2 else 'Mono'}) |\n"
            f"| Örnekleme Hızı | {seg.frame_rate} Hz |\n"
        )
    except Exception as e:
        audio_stats_md = f"⚠️ Ses istatistikleri hesaplanamadı: {e}"

    audio_file = client.files.upload(file=local_audio_path)
    while audio_file.state.name == "PROCESSING":
        time.sleep(2)
        audio_file = client.files.get(name=audio_file.name)

    progress(0.85, desc="Suno prompt üretiliyor...")
    user_prompt = f"Aşağıda bu şarkı için Fadr'dan alınan teknik veriler var. Şarkıyı dinleyip istenen formatta cevap ver.\n\n{fadr_report}"
    response = generate_with_retry(audio_file, user_prompt, SYSTEM_INSTRUCTION)
    full_text = response.text

    # ---- Sonucu bölümlere ayır: şarkı bilgisi / müzik analizi / suno prompt ----
    song_info_md = (
        f"### 🎵 {artist_name}\n"
        f"| Özellik | Değer |\n|---|---|\n"
        f"| Tempo | {tempo} BPM |\n"
        f"| Ton (Key) | {key} |\n"
        f"| Stem'ler | {', '.join(stem_types)} |\n"
        f"| Akor progresyonu | {chord_progression[:120]}{'...' if len(chord_progression) > 120 else ''} |\n"
    )

    analysis_section = full_text
    suno_prompt_text = ""
    if "=== SUNO PROMPT ===" in full_text:
        parts = full_text.split("=== SUNO PROMPT ===")
        analysis_section = parts[0].replace("=== MÜZİK ANALİZİ ===", "").strip()
        suno_prompt_text = parts[1].strip()

    analysis_md = "### 🎼 Müzik Analizi\n" + "\n".join(
        f"- **{line.split(':', 1)[0].strip()}:** {line.split(':', 1)[1].strip()}"
        if ":" in line else line
        for line in analysis_section.splitlines() if line.strip()
    )

    char_info = f"{len(suno_prompt_text)} / 1000 karakter" + (" ⚠️ SINIRI AŞIYOR" if len(suno_prompt_text) > 1000 else " ✅")

    progress(1.0, desc="Tamamlandı!")
    status_msg = f"✅ **{artist_name}** analiz edildi."
    return (
        status_msg, song_info_md, analysis_md, suno_prompt_text, char_info,
        df_chords_display, waveform_path, audio_stats_md,
        melody_paths["vocals"], melody_paths["other"], local_audio_path,
        stem_paths["vocals"], stem_paths["bass"], stem_paths["drums"],
        stem_paths["other"], stem_paths["instrumental"],
    )

# ------------------------------------------------------------
# ARAYÜZ
# ------------------------------------------------------------

THEME = gr.themes.Soft(primary_hue="violet", secondary_hue="purple")

CUSTOM_CSS = """
.gradio-container {max-width: 900px !important; margin: auto !important;}
#suno-prompt-box textarea {font-family: monospace; font-size: 14px;}
"""

with gr.Blocks(title="Suno Prompt Üretici") as app:
    gr.Markdown("## 🎧 Suno Prompt Üretici")

    with gr.Column(visible=True) as login_col:
        gr.Markdown("Bu uygulama şifre korumalı. Lütfen şifreni gir.")
        pw_input = gr.Textbox(label="Şifre", type="password")
        pw_btn = gr.Button("Giriş Yap")
        pw_error = gr.Markdown()

    with gr.Column(visible=False) as main_col:
        gr.Markdown(
            "# 🎧 Suno Prompt Üretici\n"
            "Bir şarkı yükle — teknik analiz (BPM/Key/Akor/Waveform), AI destekli derin "
            "müzikal analiz, tüm stem'ler ve **Suno için hazır prompt** otomatik üretilsin."
        )

        with gr.Row():
            song_input = gr.Audio(label="Şarkını yükle", type="filepath", scale=4)
            analyze_btn = gr.Button("🚀 Analiz Et", variant="primary", scale=1, size="lg")

        status_md = gr.Markdown(visible=True)

        with gr.Tabs():
            with gr.Tab("📊 Analiz"):
                song_info_output = gr.Markdown()
                analysis_output = gr.Markdown()

            with gr.Tab("📈 Teknik Detay"):
                waveform_output = gr.Image(label="Dalga Formu (Waveform)", show_label=True)
                audio_stats_output = gr.Markdown()
                gr.Markdown("### 🎼 Zaman Damgalı Akor Tablosu")
                chords_df_output = gr.Dataframe(
                    headers=["start", "end", "chord", "chord_basit"],
                    label="Akorlar (saniye cinsinden başlangıç/bitiş)",
                    wrap=True,
                )

            with gr.Tab("✨ Suno Prompt"):
                gr.Markdown("📋 *Metni kutunun içine tıklayıp Ctrl+A / Ctrl+C ile kopyalayabilirsin.*")
                suno_prompt_output = gr.Textbox(
                    label="Suno AI Prompt",
                    lines=6, elem_id="suno-prompt-box",
                )
                char_count_output = gr.Markdown()

            with gr.Tab("🎵 Orijinal + Melodi"):
                original_song_output = gr.Audio(label="🎵 Orijinal Şarkı (HQ)")
                gr.Markdown("---\n**Referans amaçlı sentez (MIDI'den, doğal duyulmaz):**")
                with gr.Row():
                    vocal_melody_output = gr.Audio(label="Vokal Melodisi (Sentez)")
                    instrumental_melody_output = gr.Audio(label="Enstrümantal Melodi (Sentez)")

            with gr.Tab("🎛️ Tüm Stem'ler (Gerçek Ses)"):
                gr.Markdown("Orijinal kayıttan ayrıştırılmış, tamamen doğal 5 ses katmanı:")
                with gr.Row():
                    stem_vocals_output = gr.Audio(label="🎤 Vokal")
                    stem_bass_output = gr.Audio(label="🎸 Bas")
                with gr.Row():
                    stem_drums_output = gr.Audio(label="🥁 Davul")
                    stem_other_output = gr.Audio(label="🎹 Diğer (synth/gitar/piyano)")
                stem_instrumental_output = gr.Audio(label="🎼 Tam Enstrümantal (vokal hariç hepsi)")

    all_outputs = [
        status_md, song_info_output, analysis_output, suno_prompt_output, char_count_output,
        chords_df_output, waveform_output, audio_stats_output,
        vocal_melody_output, instrumental_melody_output, original_song_output,
        stem_vocals_output, stem_bass_output, stem_drums_output,
        stem_other_output, stem_instrumental_output,
    ]

    analyze_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=all_outputs,
    ).then(
        fn=analyze_song_pipeline,
        inputs=[song_input],
        outputs=all_outputs,
    )

    def check_password(pw):
        if not APP_PASSWORD:
            return gr.update(visible=True), gr.update(visible=False), ""
        if pw == APP_PASSWORD:
            return gr.update(visible=True), gr.update(visible=False), ""
        return gr.update(visible=False), gr.update(visible=True), "❌ Yanlış şifre."

    pw_btn.click(fn=check_password, inputs=[pw_input], outputs=[main_col, login_col, pw_error])

app.queue()
app.launch(theme=THEME, css=CUSTOM_CSS, ssr_mode=False)

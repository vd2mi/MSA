# مملكة خويك

مشروع هكاثون بسيط: واجهة Flask تتصل بمحرك AI داخلي.

التشغيل المحلي:

```bash
pip install -r requirements.txt
python app.py
```

المكونات:
- `app.py` — سيرفر Flask وراوتز
- `ai_engine.py` — محرك AI (حاليًا ستب)
- `templates/` — HTML القوالب
- `static/` — CSS وJS والصور

استبدل محتوى `generate_response` في `ai_engine.py` لتوصيل موديل خارجي.

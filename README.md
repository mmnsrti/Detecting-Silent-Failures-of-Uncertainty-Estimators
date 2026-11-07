# Detecting Silent Failures of Uncertainty Estimators Under Distribution Shift

> **پروژه تحلیل آماری، نرمال‌سازی و دسته‌بندی با شبکه بیزی (BBN)**  
> **درس شناسایی الگو (Pattern Recognition)** — ارزیابی دیتاست‌های **CommonsenseQA** و **ARC-Challenge**

---

## 📋 فهرست مطالب
- [معرفی پروژه](#معرفی-پروژه)
- [ساختار مخزن (Repository Structure)](#ساختار-مخزن-repository-structure)
- [پیش‌نیازها و راه‌اندازی](#پیش‌نیازها-و-راه‌اندازی)
- [دریافت و اعتبارسنجی داده‌ها](#دریافت-و-اعتبارسنجی-داده‌ها)
- [گردش کار نوت‌بوک‌ها (Notebook Pipeline)](#گردش-کار-نوت‌بوک‌ها-notebook-pipeline)
- [اجرای خودکار و تست‌ها](#اجرای-خودکار-و-تست‌ها)
- [خروجی‌ها و نتایج کلیدی](#خروجی‌ها-و-نتایج-کلیدی)

---

## 🎯 معرفی پروژه

این پروژه بر اساس مقاله و چارچوب پژوهشی **"Detecting Silent Failures of Uncertainty Estimators"** پیاده‌سازی شده است. هدف آن بررسی تغییر توزیع (Distribution Shift)، استخراج ویژگی‌های ساختاری متن در دو دیتاست پرسش و پاسخ چندگزینه‌ای (`CommonsenseQA` و `ARC-Challenge`)، تحلیل آماری و رفتار توزیع‌ها، بررسی روش‌های سه‌گانه نرمال‌سازی داده‌ها، و در نهایت پیاده‌سازی طبقه‌بند شبکه بیزی (Bayesian Belief Network - BBN) به همراه تحلیل همبستگی پیرسون است.

---

## 📁 ساختار مخزن (Repository Structure)

```text
├── .gitignore                                      # نادیده‌گیری فایل‌های کش، محیط مجازی و متاداده‌ها
├── README.md                                       # مستندات جامع پروژه
├── requirements.txt                                # وابستگی‌های پایتون
├── Detecting Silent Failures of Uncertainty Estimators.pdf # سند مرجع مقاله / پروژه
├── download_datasets.py                            # اسکریپت دانلود و اعتبارسنجی Parquet از Hugging Face
├── execute_notebooks.py                            # اجرای خودکار ترتیبی نوت‌بوک‌ها
├── test_download_datasets.py                       # تست‌های یکپارچگی داده‌ها (Pytest)
├── test_notebooks_execution.py                     # تست‌های اعتبارسنجی خروجی‌ها و متریک‌ها (Pytest)
├── datasets/                                       # فایل‌های Parquet خام
│   ├── arc_challenge/
│   │   └── ARC-Challenge/                         # Train / Validation / Test
│   └── commonsense_qa/
│       └── data/                                  # Train / Validation / Test
├── notebooks/                                      # نوت‌بوک‌های تحلیلی ژوپیتر
│   ├── 01_prepare_numeric_features.ipynb          # گام ۱: استخراج ویژگی‌های عددی
│   ├── 02_statistics_and_distributions.ipynb      # گام ۲: تحلیل آماری، چارک‌ها و قانون تجربی
│   ├── 03_three_normalizations.ipynb              # گام ۳: نرمال‌سازی Min-Max، Z-Score و Decimal Scaling
│   ├── 04_pearson_bbn_classification.ipynb        # گام ۴: همبستگی پیرسون و طبقه‌بندی BBN
│   └── README_FA.md                               # راهنمای فارسی نوت‌بوک‌ها
└── outputs/                                        # نتایج، گزارش‌های آماری و نمودارها
    ├── dataset_manifest.csv
    ├── numeric_features.csv
    ├── descriptive_statistics.csv
    ├── quartile_intervals.csv
    ├── empirical_rule_vs_normal.csv
    ├── normalized_features.csv
    ├── normalization_parameters.csv
    ├── normalization_quality_checks.csv
    ├── charts/                                    # ۱۶ نمودار هیستوگرام، KDE و مقایسه نرمال
    └── classification/                            # گزارش‌ها، ماتریس درهم‌ریختگی و خلاصه BBN
```

---

## ⚙️ پیش‌نیازها و راه‌اندازی

برای اجرای پروژه پیشنهاد می‌شود از پایتون نسخه `3.10` یا بالاتر استفاده کنید:

```bash
# ایجاد و فعال‌سازی محیط مجازی
python -m venv .venv
# لینوکس و مک:
source .venv/bin/activate
# ویندوز:
.venv\Scripts\activate

# نصب وابستگی‌ها
pip install -r requirements.txt
```

---

## 📥 دریافت و اعتبارسنجی داده‌ها

دیتاست‌ها به صورت آفلاین داخل پوشه `datasets/` قرار دارند. در صورت نیاز به دانلود مجدد یا اعتبارسنجی سلامت فایل‌های Parquet:

```bash
python download_datasets.py
```

این اسکریپت داده‌ها را با بررسی کامل صفحات Parquet و اطمینان از عدم خرابی بارگیری و اعتبارسنجی می‌کند.

---

## 🔬 گردش کار نوت‌بوک‌ها (Notebook Pipeline)

نوت‌بوک‌ها باید به ترتیب زیر اجرا شوند:

1. **`01_prepare_numeric_features.ipynb`**:
   - خواندن داده‌های خام Parquet
   - استخراج ۴ ویژگی عددی:
     - `question_words`: تعداد کلمات صورت سؤال
     - `question_chars`: تعداد کاراکترهای سؤال
     - `avg_option_words`: میانگین طول گزینه‌ها
     - `option_count`: تعداد گزینه‌ها
   - خروجی: `outputs/numeric_features.csv` (شامل ۱۴,۶۹۲ نمونه)

2. **`02_statistics_and_distributions.ipynb`**:
   - محاسبه معیارهای مرکزی و پراکندگی (Mean, Median, Range, Standard Deviation)
   - محاسبه چارک‌های Q1، Q2، Q3 و بازه چارک چهارم (Q4: Q3 تا Max)
   - ارزیابی قانون تجربی توزیع نرمال (Empirical Rule: $68.27\%$, $95.45\%$, $99.73\%$) در بازه‌های $\mu \pm k\sigma$
   - رسم و ذخیره ۱۶ نمودار مقایسه توزیع تجربی با منحنی نرمال تئوری در `outputs/charts/`

3. **`03_three_normalizations.ipynb`**:
   - پیاده‌سازی و مقایسه ۳ روش نرمال‌سازی:
     - **Min-Max Scaling** (نگاشت به بازه $[0, 1]$)
     - **Z-Score Normalization** (میانگین ۰، انحراف معیار ۱)
     - **Decimal Scaling** ($v' = v / 10^j$)
   - برازش پارامترها صرفاً روی مجموعه داده‌های آموزشی (Train) و اعمال به Validation و Test برای جلوگیری از Data Leakage
   - ارزیابی کیفیت نرمال‌سازی و ذخیره جداول در `outputs/`

4. **`04_pearson_bbn_classification.ipynb`**:
   - محاسبه ماتریس ضریب همبستگی پیرسون (Pearson Correlation Matrix)
   - گسسته‌سازی ویژگی‌ها بر اساس صدک‌ها
   - پیاده‌سازی کامل شبکه بیزی (Bayesian Belief Network) با برآورد احتمال شرطی و هموارسازی لاپلاس (Laplace Smoothing)
   - پیش‌بینی و دسته‌بندی منشأ نمونه‌ها روی داده‌های Test
   - رسم Confusion Matrix و تولید گزارش طبقه‌بندی در `outputs/classification/`

---

## 🧪 اجرای خودکار و تست‌ها

تمامی مراحل و خروجی‌های پروژه از طریق مجموعه تست‌های خودکار پیاده‌سازی شده در `pytest` پوشش داده شده‌اند:

```bash
# اجرای تمامی تست‌ها
pytest -v
```

همچنین برای اجرای کامل و بدون سرپرست تمامی نوت‌بوک‌ها و به‌روزرسانی خروجی‌ها:

```bash
python execute_notebooks.py
```

---

## 📊 خروجی‌ها و نتایج کلیدی

- **تعداد کل نمونه‌ها**: ۱۴,۶۹۲ نمونه (۱۲,۱۰۲ نمونه از CommonsenseQA و ۲,۵۹۰ نمونه از ARC-Challenge)
- **دقت شبکه بیزی (BBN Accuracy)**: بالای **۶۵٪** روی داده‌های تست برای تفکیک دو دیتاست بر مبنای ویژگی‌های ساختاری متن
- **نمودارها و گزارش‌ها**: تمام ماتریس‌های درهم‌ریختگی، ضرایب همبستگی و نمودارهای توزیع به صورت خودکار در پوشه `outputs/` و به صورت فشرده در `outputs.zip` ذخیره شده‌اند.

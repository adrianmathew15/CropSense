# CropSense

CropSense is a Flask-based agriculture assistant with two local web applications:

- Main CropSense app on port `5000`: crop recommendation, fertilizer recommendation, crop disease prediction, and nutrient deficiency detection.
- Precision agriculture WebApp on port `8080`: farmland registration, polygon-based field selection, vegetation index maps, fertilizer maps, and crop-field data management using Google Earth Engine.

The main app can launch the WebApp link from its navigation, but the WebApp should be run in its own terminal for the most reliable local setup.

## Prerequisites

- Python 3.10 is recommended.
- A valid OpenWeather API key for weather-based crop recommendation.
- A Google account with Earth Engine access enabled.
- A Google Cloud project registered for Earth Engine.

## Project Structure

```text
CropSense/
  app.py                  Main CropSense Flask app
  config.py               Main app configuration, including OpenWeather key
  requirements.txt        Main app dependencies
  webapp/webapp/          Precision agriculture WebApp
    run.py                WebApp Flask runner
    config.py             WebApp configuration, including EE_PROJECT
    requirements.txt      WebApp dependencies
```

## 1. Set Up the Main CropSense App

Clone the repository and enter the project folder:

```powershell
git clone <repository-url>
cd CropSense
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

On macOS/Linux, use:

```bash
git clone <repository-url>
cd CropSense
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Add your OpenWeather API key in `config.py`:

```python
weather_api_key = "YOUR_OPENWEATHER_API_KEY"
```

Run the main app:

```powershell
.\.venv\Scripts\python.exe app.py
```

On macOS/Linux:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/home
```

## 2. Get an OpenWeather API Key

1. Go to https://openweathermap.org/api.
2. Sign up or log in.
3. Open your account dashboard.
4. Go to the `API keys` tab.
5. Copy an existing key or create a new one.
6. Paste it into the root `config.py`.

You can test the key in a browser:

```text
https://api.openweathermap.org/data/2.5/weather?q=London&appid=YOUR_OPENWEATHER_API_KEY
```

If the response contains a `main` object, the key is working. New keys can take a few minutes to activate.

Note: Crop prediction can also use manually entered temperature and humidity, so the app can still run even if OpenWeather is not configured.

## 3. Set Up the Precision Agriculture WebApp

Open a second terminal:

```powershell
cd CropSense\webapp\webapp
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

On macOS/Linux:

```bash
cd CropSense/webapp/webapp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the WebApp:

```powershell
.\.venv\Scripts\python.exe run.py
```

On macOS/Linux:

```bash
python run.py
```

Open:

```text
http://127.0.0.1:8080
```

## 4. Google Earth Engine Setup

The WebApp uses Google Earth Engine for vegetation and fertilizer map generation.

### Create or Choose a Google Cloud Project

In the Google Cloud Console project selector, copy the project ID. Example:

```text
my-earthengine-project
```

Then register that project for Earth Engine:

```text
https://console.cloud.google.com/earth-engine/configuration?project=YOUR_PROJECT_ID
```

Make sure the Earth Engine API is enabled for that project.

### Configure the WebApp Project ID

In `webapp/webapp/.env`, set:

```env
EE_PROJECT=YOUR_PROJECT_ID
```

You can also set it directly in `webapp/webapp/config.py`, but `.env` is preferred for local configuration.

### Authenticate Earth Engine

From the WebApp folder:

```powershell
cd CropSense\webapp\webapp
.\.venv\Scripts\earthengine.exe authenticate
.\.venv\Scripts\earthengine.exe set_project YOUR_PROJECT_ID
```

On macOS/Linux:

```bash
cd CropSense/webapp/webapp
source .venv/bin/activate
earthengine authenticate
earthengine set_project YOUR_PROJECT_ID
```

Verify the connection:

```powershell
.\.venv\Scripts\python.exe -c "import ee; ee.Initialize(project='YOUR_PROJECT_ID'); print(ee.Number(1).getInfo())"
```

On macOS/Linux:

```bash
python -c "import ee; ee.Initialize(project='YOUR_PROJECT_ID'); print(ee.Number(1).getInfo())"
```

Expected output:

```text
1
```

If you see `Project ... is not registered to use Earth Engine`, register the project using the Earth Engine configuration link above and try again.

## 5. Running Both Apps Together

Terminal 1:

```powershell
cd CropSense
.\.venv\Scripts\python.exe app.py
```

Terminal 2:

```powershell
cd CropSense\webapp\webapp
.\.venv\Scripts\python.exe run.py
```

On macOS/Linux, use the equivalent paths and activate each virtual environment before running:

```bash
cd CropSense
source .venv/bin/activate
python app.py
```

```bash
cd CropSense/webapp/webapp
source .venv/bin/activate
python run.py
```

Use:

```text
Main app: http://127.0.0.1:5000/home
WebApp:   http://127.0.0.1:8080
```

## Common Issues

### OpenWeather says `Invalid API key`

Generate a new key from the OpenWeather dashboard, paste it into root `config.py`, restart the main app, and wait a few minutes if the key was just created.

### Earth Engine says the project is not registered

Open:

```text
https://console.cloud.google.com/earth-engine/configuration?project=YOUR_PROJECT_ID
```

Register the project, enable Earth Engine, then rerun:

```powershell
.\.venv\Scripts\earthengine.exe set_project YOUR_PROJECT_ID
```

### WebApp maps are blank

Check that:

- The WebApp was restarted after configuration changes.
- `EE_PROJECT` is set correctly.
- Earth Engine authentication succeeds.
- The selected farmland has a valid polygon.

### Main app WebApp link shows a missing `run.py`

Run the WebApp manually in a second terminal from:

```powershell
cd CropSense\webapp\webapp
.\.venv\Scripts\python.exe run.py
```

On macOS/Linux:

```bash
cd CropSense/webapp/webapp
source .venv/bin/activate
python run.py
```

## Notes

- The two apps use separate virtual environments and separate dependency files.
- The nested WebApp has its own README in `webapp/webapp/README.md`, but this root README contains the recommended local run process for the full CropSense project.

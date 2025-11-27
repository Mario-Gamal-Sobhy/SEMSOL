# ☁️ Deployment Guide: Streamlit Cloud + MongoDB Atlas

This guide will help you deploy the SEMSOL Engagement Monitoring System to Streamlit Cloud.

## 📋 Prerequisites

1.  **GitHub Account**: To host your code.
2.  **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io/).
3.  **MongoDB Atlas Account**: Free cloud database.

---

## 🚀 Step 1: Set Up MongoDB Atlas (Cloud Database)

Since Streamlit Cloud cannot access your local MongoDB, you need a cloud database.

1.  **Create Account**: Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) and sign up (Free).
2.  **Create Cluster**:
    *   Build a Database.
    *   Choose **M0 (Free)** tier.
    *   Select a provider/region (e.g., AWS / N. Virginia).
    *   Click **Create**.
3.  **Create User**:
    *   Go to **Database Access** (sidebar).
    *   Add New Database User.
    *   Username: `semsol_user` (example).
    *   Password: `your_secure_password` (Save this!).
    *   Role: Read and write to any database.
4.  **Allow Access**:
    *   Go to **Network Access** (sidebar).
    *   Add IP Address.
    *   Select **Allow Access from Anywhere** (`0.0.0.0/0`). (Required for Streamlit Cloud).
    *   Confirm.
5.  **Get Connection String**:
    *   Go to **Database** (sidebar) -> Click **Connect**.
    *   Choose **Drivers** (Python).
    *   Copy the connection string. It looks like:
        `mongodb+srv://semsol_user:<password>@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority`
    *   **Replace `<password>`** with your actual password.

---

## 📦 Step 2: Prepare Your Code

1.  **Push to GitHub**:
    Ensure your project is pushed to a GitHub repository.
    *   Make sure `requirements_deploy.txt` is in the root.
    *   Make sure `app_deploy.py` is in the root.

---

## ☁️ Step 3: Deploy to Streamlit Cloud

1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **New App**.
3.  **Repository**: Select your GitHub repo (`S.E.M.S.O.L`).
4.  **Branch**: `main` (or your working branch).
5.  **Main file path**: `app_deploy.py` (NOT `app.py`).
    *   *Note: `app.py` won't work because of webcam issues on cloud.*
6.  Click **Deploy!**

---

## 🔐 Step 4: Configure Secrets (Connect Database)

Your app will fail to connect to MongoDB initially. You need to add the secrets.

1.  In your deployed app dashboard, click **Manage app** (bottom right) -> **Settings** (three dots).
2.  Go to **Secrets**.
3.  Paste the following (using YOUR connection string from Step 1):

```toml
[mongo]
connection_string = "mongodb+srv://semsol_user:your_password@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority"
database_name = "semsol_engagement"
```

4.  Click **Save**.
5.  The app should restart automatically and connect!

---

## 🧪 Verification

1.  Open your deployed app URL.
2.  Grant camera permissions (browser).
3.  You should see the video feed.
4.  Start monitoring.
5.  Stop monitoring -> Check if it says "Saved to MongoDB".
6.  Check your MongoDB Atlas collection to see the data!

---

## ❓ Troubleshooting

*   **"ModuleNotFoundError"**: Check `requirements_deploy.txt`.
*   **Camera not working**: Ensure you are using `app_deploy.py` and have granted browser permissions.
*   **Database error**: Check your connection string in Secrets. Did you replace `<password>`? Did you allow IP `0.0.0.0/0`?

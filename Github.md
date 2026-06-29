# GitHub Desktop + CLI Quick Setup Guide

A minimal, follow-along guide for Windows & macOS.

---

## 1. Install GitHub Desktop

### Windows
1. Go to **https://desktop.github.com**
2. Click **Download for Windows** → run the `.exe`
3. It installs and launches automatically.

### macOS
1. Go to **https://desktop.github.com**
2. Click **Download for macOS** → open the `.zip`
3. Drag **GitHub Desktop** into **Applications**, then open it.

> Or install via package manager:
> - **Windows:** `winget install --id GitHub.GitHubDesktop`
> - **macOS:** `brew install --cask github`

### Sign in
1. **File → Options** (Win) / **GitHub Desktop → Settings** (Mac) → **Accounts**
2. Click **Sign in to GitHub.com** → authorize in browser.
3. Set your name & email under **Git** tab (used for commits).

---

## 2. Create a Private Repo (on the Web)

1. Go to **https://github.com** and sign in.
2. Click the **`+`** (top-right) → **New repository**.
3. Fill in:
   - **Repository name:** `my-project`
   - **Visibility:** select **Private** ✅
   - (Optional) check **Add a README file**
4. Click **Create repository**.

---

## 3. Clone the Repo in GitHub Desktop

1. On the repo page, click the green **Code** button.
2. Choose **Open with GitHub Desktop**.
3. GitHub Desktop opens → confirm the **local path** → click **Clone**.

> **Alternative inside the app:** **File → Clone repository → GitHub.com tab → select the repo → Clone.**

You can now edit files locally, then **commit** and **push** from the Desktop app.

---

## 4. Set Up GitHub CLI (`gh`)

### Install

| OS | Command |
|----|---------|
| **Windows** | `winget install --id GitHub.cli` |
| **macOS** | `brew install gh` |

> Verify: `gh --version`

### Authenticate
```bash
gh auth login
```
Answer the prompts:
- **Account:** GitHub.com
- **Protocol:** HTTPS
- **Authenticate:** Login with a web browser → copy the one-time code → paste in browser.

> Check status anytime: `gh auth status`

---

## 5. Handy `gh` Commands

```bash
# Create a private repo from the current folder and push it
gh repo create my-project --private --source=. --remote=origin --push

# Create an empty private repo on GitHub
gh repo create my-project --private

# Clone a repo
gh repo clone username/my-project

# List your repos
gh repo list

# View a repo in the browser
gh repo view --web
```

---

## Quick Reference

| Task | Tool | Action |
|------|------|--------|
| Install Desktop | Desktop | download from desktop.github.com |
| Create private repo | Web | `+` → New repository → Private |
| Clone repo | Desktop | Code → Open with GitHub Desktop |
| Install CLI | Terminal | `winget install GitHub.cli` / `brew install gh` |
| Login CLI | Terminal | `gh auth login` |
| Push folder to new repo | CLI | `gh repo create NAME --private --source=. --push` |

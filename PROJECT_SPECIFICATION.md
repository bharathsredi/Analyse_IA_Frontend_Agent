# PROJECT: Analyse_IA - Complete Frontend Implementation

## TECH STACK:
- Framework: Next.js 16.1.6 (App Router)
- Language: TypeScript
- Styling: Tailwind CSS 3.3.0
- State Management: Zustand 4.4.7
- HTTP Client: Axios 1.6.5
- Charts: @tremor/react 3.14.1
- Icons: lucide-react 0.309.0
- Data Fetching: @tanstack/react-query 5.17.19

## BACKEND API ENDPOINTS (Running on http://localhost:8000):
- `POST /auth/register` - Register user
- `POST /auth/login` - Login user
- `POST /auth/refresh` - Refresh JWT token
- `GET /files/list` - List uploaded files
- `POST /files/upload/csv` - Upload CSV (max 50MB)
- `POST /files/upload/pdf` - Upload PDF (max 20MB)
- `POST /agent/ask` - Submit query (returns task_id, 202 Accepted)
- `GET /agent/status/{task_id}` - Check task status (returns status: PENDING/SUCCESS/FAILURE, result: BackendResult)

---

## PHASE 1: AUTHENTICATION PAGES

### 1.1 Register Page (`/app/(auth)/register/page.tsx`)
- Full name, email (regex validation), password (show/hide), confirm password.
- Password strength meter (length, uppercase, lowercase, numbers, symbols).
- Language toggle (FR/EN) - persistent in store.
- Auto-redirect to dashboard after 800ms delay on success.
- Styling: Dark theme (`bg-[#0A1628]`, `text-white`, blue accents).

### 1.2 Login Page (`/app/(auth)/login/page.tsx`)
- Email, password (show/hide), "Remember me" checkbox.
- Error handling from backend, loading spinner on submit.
- Auto-redirect to dashboard on success.

### 1.3 Auth Flow Implementation
- JWT storage in localStorage (`access_token`, `refresh_token`).
- Auto-token refresh via Axios interceptors on 401 responses.
- Protected routes (redirect to login if no token).

---

## PHASE 2: DASHBOARD LAYOUT

### 2.1 Dashboard Layout (`/app/(dashboard)/layout.tsx`)
- Sidebar navigation (desktop 250px) / hamburger drawer (mobile).
- Items: Dashboard, Upload, History, RGPD, Settings.
- User profile section at bottom (Email, Language toggle, Logout).
- Styling: Sidebar bg `rgba(15, 31, 61, 0.8)`, active link highlight `bg-blue-600/15`.

---

## PHASE 3: CHAT & DASHBOARD

### 3.1 Dashboard Chat Page (`/app/(dashboard)/dashboard/page.tsx`)
- **Empty State:** Bot icon, 4 clickable prompt suggestions in FR/EN.
- **Message Display:** User messages right-aligned (blue gradient). Assistant left-aligned (dark bg). Streaming animation while waiting.
- **File Selection:** Paperclip icon to open drawer. Select from uploaded files or auto-select most recent CSV.
- **Input Area:** Auto-grow textarea, Send button, disabled during processing.
- **Task Polling:** Send query -> Get `task_id` -> Poll `/agent/status/{task_id}` every 3 seconds (max 60s timeout).
- **Results:** Render `MetricsBar`, `ShapChart`, and `AnomalyTable` on SUCCESS.

---

## PHASE 4: FILE UPLOAD

### 4.1 Upload Page (`/app/(dashboard)/upload/page.tsx`)
- Separate drag-and-drop zones for CSV (50MB) and PDF (20MB).
- Upload progress bar (0-100%).
- Uploaded Files List: Table showing file type, name, size, upload date, and a delete button.

---

## PHASE 5: HISTORY PAGE

### 5.1 History Page (`/app/(dashboard)/history/page.tsx`)
- Display past queries (text, response summary, timestamp).
- Sort newest first, filter/search capabilities.
- Click to view full result or re-run query.

---

## PHASE 6: RGPD PAGE

### 6.1 RGPD Page (`/app/(dashboard)/rgpd/page.tsx`)
- **Data Export:** Download JSON file of user data.
- **Account Deletion:** Red warning, password confirmation modal, hits `DELETE /auth/delete-account`.
- **Consent Management:** Checkboxes for analytics/tracking.

---

## PHASE 7: COMPONENTS & UTILITIES

- **Create/Verify Components:** `Toast`, `Modal`, `Sidebar`, `LoadingSpinner`, `PasswordStrengthMeter`, `AnalysisVisualization`.
- **Utilities (`/lib/`):** `api.ts` (Axios auth/refresh logic), `auth.ts`, `i18n.ts`, `utils.ts` (formatting).
- **Zustand Stores (`/store/`):** - `authStore.ts` (isAuthenticated, user state).
  - `chatStore.ts` (messages, language, isProcessing, currentTaskId).

---

## PHASE 8: TYPES & INTERFACES

Create `/types/index.ts` containing strict TypeScript interfaces for:
- `ChatMessage` (role, content, language, timestamp, analysis result)
- `UploadedFile` (file_id, filename, size, type)
- `AnalysisResult` and `BackendResult`
- `TopFeature` and `Anomaly`

---

## PHASE 9: STYLING & THEME
- **Palette:** Primary Dark `#0A1628`, Secondary Dark `#0F1F3D`, Accent Blue `#2563EB`.
- Global dark mode setup via Tailwind, smooth transitions.

---

## PHASE 10: INTERNATIONALIZATION (i18n)
- Implement translation keys for Auth, Chat, Upload, and RGPD sections.
- Ensure language toggle immediately updates UI without page reload.
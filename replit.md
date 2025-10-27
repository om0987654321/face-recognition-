# Face Recognition Attendance System

## Overview

This is an AI-powered attendance management system for educational institutions that uses facial recognition technology to automate student attendance tracking. The system allows administrators to register students with their face photos, train a machine learning model, and mark attendance automatically by recognizing students through a webcam.

The application provides a complete workflow from student registration through face capture, model training, and real-time attendance marking with visual analytics and reporting capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Framework
**Problem:** Need a rapid development framework for an interactive web application with real-time camera access and data visualization.

**Solution:** Streamlit-based web application (`app.py`)

**Rationale:** Streamlit provides a Python-native way to build interactive web applications with minimal frontend code, native support for camera access, and easy integration with data science libraries. This eliminates the need for separate frontend/backend architecture while maintaining a modern, responsive UI.

**Pros:**
- Rapid development with pure Python
- Built-in widgets for forms, file uploads, and camera access
- Automatic reactivity and state management
- Native integration with visualization libraries (Plotly)

**Cons:**
- Less flexibility for complex custom UI components
- Limited control over client-side behavior

### Face Detection & Recognition
**Problem:** Need accurate, fast face detection and recognition that works in various lighting conditions and angles.

**Solution:** Two-stage approach using MediaPipe for detection and Random Forest for classification

**Components:**
1. **MediaPipe Face Detection** - Detects faces in camera feed with bounding boxes
2. **Face Embedding** - Converts 64x64 grayscale face crops to flattened feature vectors (4096 dimensions, normalized 0-1)
3. **Random Forest Classifier** - Trained on embeddings to identify students with confidence scores

**Rationale:** MediaPipe provides robust, production-ready face detection with minimal computational overhead. Random Forest offers good accuracy with relatively small training datasets and handles multi-class classification naturally with built-in probability estimates.

**Threshold:** 60% confidence minimum to prevent false positives

**Pros:**
- MediaPipe is lightweight and runs efficiently on CPU
- No need for GPU or deep learning infrastructure
- Random Forest trains quickly even with limited data
- Explainable predictions with confidence scores

**Cons:**
- Less accurate than deep learning approaches (e.g., FaceNet, ArcFace)
- Requires consistent face angles and lighting for best results
- 32x32 or 64x64 embeddings are relatively low-resolution features

### Data Storage
**Problem:** Need persistent storage for student records, face images, attendance logs, and trained models.

**Solution:** SQLite database with file-based image storage

**Schema:**
- `students` table: id, name, roll_number, class, section, registration_number, created_at
- `attendance` table: id, student_id, name, timestamp (foreign key to students)

**Face Images:** Stored as individual files in `dataset/{student_id}/` directories

**Model Storage:** Serialized scikit-learn model saved as `face_model.pkl` using pickle

**Rationale:** SQLite provides zero-configuration embedded database perfect for single-instance applications. File-based image storage avoids BLOB complexity and makes the dataset easily inspectable and portable.

**Pros:**
- No database server setup required
- Self-contained database file
- ACID transactions for data integrity
- Easy backup and portability

**Cons:**
- Not suitable for concurrent multi-user access
- No built-in replication or clustering
- File-based images less efficient than object storage at scale

### Student Registration Workflow
**Problem:** Need to capture multiple face angles during registration for robust model training.

**Solution:** Automated multi-photo capture (5-15 images) with real-time preview

**Process:**
1. User enters student details (name, roll number, class, section, registration number)
2. Camera opens with live preview
3. System automatically captures multiple photos with small time intervals
4. Each photo is processed through MediaPipe to verify face detection
5. Valid face crops are saved to `dataset/{student_id}/image_{index}.jpg`
6. Student record created in database with timestamp

**Rationale:** Multiple photos from different angles improve model accuracy by providing diverse training samples. Automatic capture ensures consistent timing and reduces user error.

### Model Training
**Problem:** Need to retrain recognition model as new students are registered.

**Solution:** Background training process with progress tracking

**Process:**
1. Load all images from `dataset/` directory structure
2. Extract embeddings for each image using MediaPipe detection
3. Build training dataset (X=embeddings, y=student_ids)
4. Train Random Forest classifier with default parameters
5. Serialize model to `face_model.pkl`
6. Update UI with training progress and completion status

**Rationale:** Background processing prevents UI blocking during training. Progress updates provide user feedback for long-running operations.

### Attendance Marking
**Problem:** Need real-time face recognition with duplicate prevention.

**Solution:** Live camera feed with instant recognition and same-day duplicate checking

**Process:**
1. Open camera and display live preview
2. Capture frame and detect faces with MediaPipe
3. Extract embedding and run through trained model
4. If confidence > 60%, identify student
5. Check database for existing attendance record on current date
6. If not duplicate, insert attendance record with timestamp
7. Display confirmation with student name, confidence, and time

**Validation Rules:**
- Minimum 60% confidence threshold
- One attendance record per student per day
- Face must be detected in frame

### Analytics & Reporting
**Problem:** Need visual insights into attendance patterns and student participation.

**Solution:** Plotly-based interactive charts and statistics

**Features:**
1. **Dashboard Metrics:** Total students, today's attendance count, attendance percentage
2. **Attendance Trends:** Line/bar charts showing daily attendance over time
3. **Class-wise Analysis:** Distribution of attendance by class/section
4. **Student Records:** Searchable table with attendance history per student
5. **Export Capability:** Download attendance data as CSV/Excel

**Rationale:** Plotly provides interactive, professional visualizations with minimal code. Pandas enables efficient data aggregation and filtering.

## External Dependencies

### Core Libraries
- **Streamlit (latest)** - Web application framework and UI components
- **OpenCV (opencv-python-headless)** - Image processing and camera access
- **MediaPipe** - Google's face detection solution
- **scikit-learn** - Random Forest classifier and ML utilities
- **Pandas** - Data manipulation and CSV export
- **NumPy** - Numerical operations and array handling
- **Pillow (PIL)** - Image format conversions
- **Plotly** - Interactive data visualizations

### Database
- **SQLite3** (Python standard library) - Embedded relational database

### File Storage
- Local filesystem for face images (`dataset/` directory)
- Local filesystem for trained model (`face_model.pkl`)
- Local database file (`attendance.db`)

### Browser Requirements
- Modern web browser with webcam support
- JavaScript enabled for Streamlit interactivity
- Camera permissions granted

### System Requirements
- Python 3.11+
- Webcam or camera device
- Sufficient disk space for face images (~100KB per student)

### Notes on Alternative Implementations
The `attached_assets/` directory contains Flask-based implementations with both SQLite and MySQL support, indicating the system was originally designed with Flask but migrated to Streamlit. The current production version uses Streamlit with SQLite as documented above.
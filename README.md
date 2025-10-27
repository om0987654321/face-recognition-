# 🎓 Face Recognition Attendance System

A modern, AI-powered attendance management system for schools and colleges using facial recognition technology. Built with Python, Streamlit, MediaPipe, and Machine Learning.

## ✨ Features

### 🎯 Core Functionality
- **Student Registration** - Register students with personal details and face photos
- **Bulk Import** - Import multiple students at once via CSV file upload
- **Face Recognition** - Automatic student identification using webcam
- **Attendance Marking** - Real-time attendance marking with face detection
- **Manual Entry** - Override attendance with manual entry when needed
- **Camera Selection** - Choose from multiple camera sources with testing
- **Dashboard Analytics** - Visual statistics and attendance trends
- **Advanced Analytics** - Detailed trends, patterns, and student performance tracking
- **Class Reports** - Generate class-wise and section-wise attendance reports
- **Student Management** - View, search, and manage student records
- **Attendance Records** - Filter and export attendance data

### 🎨 Modern Design
- Beautiful gradient-based UI with organized sidebar navigation
- Responsive layout for all screen sizes
- Interactive charts and visualizations (line, bar, area charts)
- Real-time camera preview with multiple camera support
- Professional purple-blue gradient color scheme
- Tabbed interfaces for complex features

### 🔒 Technology Stack
- **Frontend:** Streamlit
- **Face Detection:** MediaPipe
- **Machine Learning:** scikit-learn (Random Forest)
- **Database:** SQLite
- **Image Processing:** OpenCV
- **Visualization:** Plotly
- **Data Processing:** Pandas, NumPy

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- Webcam/Camera access
- Modern web browser

### Installation

1. **Install Dependencies**
   All required packages are already configured in this environment:
   - streamlit
   - opencv-python-headless
   - mediapipe
   - scikit-learn
   - pandas
   - numpy
   - Pillow
   - plotly

2. **Run the Application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

3. **Access the System**
   Open your browser and navigate to `http://localhost:5000`

## 📖 How to Use

### 1️⃣ Register Students

**Option A: Individual Registration**
1. Click **"Register Student"** from the sidebar
2. Fill in student details (name, roll number, class, section)
3. Click **"Register Student"** and allow camera access
4. The system will automatically capture 10 face photos
5. Student will be added to the database

**Option B: Bulk Import (NEW!)**
1. Click **"Bulk Import"** from the sidebar
2. Download the CSV template
3. Fill in student details (only name is required)
4. Upload your CSV file
5. Review and import - the system will validate and import all students

### 2️⃣ Train the Model

1. After registering students, click **"Train Model"** from sidebar
2. Click **"Start Training"** button
3. Wait for the training to complete (shows progress)
4. Model is now ready for face recognition

### 3️⃣ Mark Attendance

**Option A: Automatic Face Recognition**
1. Click **"Mark Attendance"** from sidebar
2. Click **"Open Camera"** button
3. Show your face to the camera
4. System will automatically recognize and mark attendance
5. Attendance record is saved with timestamp

**Option B: Manual Entry (NEW!)**
1. Click **"Manual Entry"** from sidebar
2. Select student from dropdown
3. Click **"Mark Present"** to manually mark attendance
4. Useful when face recognition is unavailable

### 4️⃣ View Reports & Analytics

**Basic Records**
1. Click **"View Records"** from sidebar
2. Filter by period (Today, This Week, This Month, All Time)
3. View attendance statistics and records
4. Download CSV reports

**Advanced Analytics (NEW!)**
1. Click **"Analytics"** from sidebar
2. Explore three tabs:
   - **Trends**: View attendance patterns with multiple chart types
   - **Student Performance**: Top performers and individual statistics
   - **Class Analysis**: Class-wise and section-wise breakdown
3. Interactive charts with hover details and metrics

**Class Reports (NEW!)**
1. Click **"Class Reports"** from sidebar
2. Filter by specific class and section
3. Select time period
4. View detailed metrics and download custom reports

### 5️⃣ Camera Settings (NEW!)

1. Click **"Manual Entry"** > **"Camera Settings"** tab
2. Select from available camera sources
3. Test camera to verify it works
4. Save your preferred camera selection

### 6️⃣ Manage Students

1. Click **"Manage Students"** from sidebar
2. Search for students by name or roll number
3. View student details
4. Delete students if needed

## 📊 Dashboard Features

### Statistics Cards
- **Total Students** - Number of registered students
- **Today's Attendance** - Students who marked attendance today
- **Attendance Rate** - Percentage of students present

### Attendance Trends
- Visual chart showing 30-day attendance history
- Interactive hover to see exact counts
- Helps identify patterns and trends

### Quick Actions
- Quick access to common tasks
- One-click navigation
- Streamlined workflow

## 🔧 Technical Details

### Face Recognition Process

1. **Capture** - Webcam captures student face images
2. **Detection** - MediaPipe detects face in the image
3. **Embedding** - Face is converted to a 64x64 grayscale embedding
4. **Training** - Random Forest classifier learns student faces
5. **Recognition** - New faces are matched with confidence score
6. **Threshold** - Only matches above 60% confidence are accepted

### Database Schema

**Students Table:**
- id (Primary Key)
- name
- roll_number
- class
- section
- registration_number
- created_at

**Attendance Table:**
- id (Primary Key)
- student_id (Foreign Key)
- name
- timestamp

### File Structure

```
.
├── app.py                          # Main Streamlit application
├── database.py                     # Database operations
├── face_recognition_model.py       # ML model and face recognition
├── README.md                       # This file
├── attendance.db                   # SQLite database (auto-created)
├── face_model.pkl                  # Trained ML model (auto-created)
└── dataset/                        # Student face images (auto-created)
    └── [student_id]/
        ├── face_0.jpg
        ├── face_1.jpg
        └── ...
```

## 🎯 Best Practices

### For Best Results:
- **Good Lighting** - Ensure adequate lighting when capturing faces
- **Face Position** - Look directly at camera, avoid extreme angles
- **Multiple Photos** - Capture 10+ photos with slight variations
- **Regular Training** - Retrain model after adding new students
- **Clean Background** - Use plain backgrounds when possible

### Performance Tips:
- Train model with at least 5-10 photos per student
- Retrain model periodically for better accuracy
- Ensure camera is at eye level
- Maintain consistent lighting conditions
- Keep face images updated

## ⚠️ Troubleshooting

### Camera Not Working
- Check if camera is connected and working
- Allow browser to access camera
- Close other applications using camera
- Restart the application

### Face Not Detected
- Ensure adequate lighting
- Position face in center of frame
- Remove obstructions (glasses, mask if possible)
- Move closer to camera

### Low Recognition Accuracy
- Retrain the model with more photos
- Ensure training photos are clear
- Use consistent lighting
- Capture photos from different angles

### Database Issues
- Check if attendance.db file exists
- Ensure write permissions
- Restart application if needed

## 🌟 Features Roadmap

Future enhancements planned:
- Multi-camera support
- Email/SMS notifications
- Advanced analytics and reports
- Class-wise attendance tracking
- Export to Excel with formatting
- Attendance scheduling
- Parent portal access
- Mobile app integration

## 📝 Notes

- All data is stored locally in SQLite database
- Face images are stored in the `dataset` folder
- Model file (`face_model.pkl`) contains trained recognition data
- System works completely offline after setup
- No cloud services or internet required for operation

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify camera and browser permissions
3. Ensure all dependencies are installed
4. Review console logs for error messages

## 📄 License

This is an educational project for learning and demonstration purposes.

---

**Made with ❤️ for Educational Institutions**

Empowering schools and colleges with modern attendance management!

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import time
from datetime import datetime, timedelta

from database import (
    init_database, add_student, get_all_students, get_student_by_id,
    delete_student, mark_attendance, get_attendance_records,
    get_attendance_stats, get_total_students, get_today_attendance_count,
    bulk_import_students, get_class_wise_attendance, get_student_attendance_summary
)
from face_recognition_model import (
    save_face_image, train_model, predict_face, is_model_trained,
    delete_student_images, extract_face_embedding
)

st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    """Apply custom CSS for modern, colorful design"""
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .big-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #555;
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            color: white;
            text-align: center;
            margin: 1rem 0;
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .stat-label {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .success-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        
        .error-box {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        
        .info-box {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        
        .card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }
        
        h1, h2, h3 {
            color: #667eea;
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        </style>
    """, unsafe_allow_html=True)

def capture_from_camera(num_images=10):
    """Capture multiple images from webcam"""
    cap = cv2.VideoCapture(0)
    images = []
    
    camera_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if not cap.isOpened():
        st.error("Could not access camera. Please check your webcam connection.")
        return images
    
    st.info(f"📸 Capturing {num_images} images. Please look at the camera and move slightly between captures.")
    time.sleep(2)
    
    for i in range(num_images):
        ret, frame = cap.read()
        if ret:
            images.append(frame.copy())
            
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(display_frame, channels="RGB", use_container_width=True)
            
            progress = (i + 1) / num_images
            progress_bar.progress(progress)
            status_text.text(f"Captured {i + 1}/{num_images} images")
            
            time.sleep(0.3)
    
    cap.release()
    camera_placeholder.empty()
    progress_bar.empty()
    status_text.empty()
    
    return images

def home_page():
    """Home page with dashboard"""
    st.markdown('<h1 class="big-title">🎓 Face Recognition Attendance System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Modern AI-Powered Attendance Management for Educational Institutions</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    total_students = get_total_students()
    today_attendance = get_today_attendance_count()
    attendance_rate = (today_attendance / total_students * 100) if total_students > 0 else 0
    
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">👥 Total Students</div>
                <div class="stat-number">{total_students}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">✅ Today's Attendance</div>
                <div class="stat-number">{today_attendance}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">📊 Attendance Rate</div>
                <div class="stat-number">{attendance_rate:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📈 Attendance Trends (Last 30 Days)")
    dates, counts = get_attendance_stats(30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=counts,
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title="Daily Attendance Count",
        xaxis_title="Date",
        yaxis_title="Number of Students",
        hovermode='x unified',
        plot_bgcolor='white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🎯 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Register New Student", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()
    
    with col2:
        if st.button("📸 Mark Attendance", use_container_width=True):
            st.session_state.page = "mark_attendance"
            st.rerun()
    
    with col3:
        if st.button("📋 View Records", use_container_width=True):
            st.session_state.page = "records"
            st.rerun()

def register_student_page():
    """Student registration page"""
    st.markdown('<h1 class="big-title">➕ Register New Student</h1>', unsafe_allow_html=True)
    
    with st.form("student_registration"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("👤 Full Name *", placeholder="Enter student's full name")
            roll_number = st.text_input("🔢 Roll Number", placeholder="e.g., 2024001")
            class_name = st.text_input("📚 Class", placeholder="e.g., 10th Grade")
        
        with col2:
            section = st.text_input("📑 Section", placeholder="e.g., A")
            registration_number = st.text_input("📄 Registration Number", placeholder="e.g., REG2024001")
        
        st.markdown("---")
        st.markdown("### 📸 Capture Face Photos")
        st.info("Click 'Capture Photos' to take 10 photos for training. Please look at the camera and move slightly between captures.")
        
        num_photos = st.slider("Number of photos to capture", 5, 15, 10)
        
        submitted = st.form_submit_button("🚀 Register Student", use_container_width=True)
    
    if submitted:
        if not name:
            st.error("❌ Please enter student's name!")
            return
        
        with st.spinner("📸 Starting camera..."):
            images = capture_from_camera(num_photos)
        
        if len(images) == 0:
            st.error("❌ Failed to capture images. Please check your camera.")
            return
        
        with st.spinner("💾 Saving student information..."):
            student_id = add_student(name, roll_number, class_name, section, registration_number)
            
            valid_images = 0
            for idx, img in enumerate(images):
                embedding = extract_face_embedding(img)
                if embedding is not None:
                    save_face_image(student_id, img, idx)
                    valid_images += 1
            
            if valid_images == 0:
                delete_student(student_id)
                st.error("❌ No face detected in the captured images. Please try again.")
                return
            
            st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Student Registered Successfully!</h3>
                    <p><strong>Name:</strong> {name}</p>
                    <p><strong>Student ID:</strong> {student_id}</p>
                    <p><strong>Photos Captured:</strong> {valid_images}/{len(images)}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.warning("⚠️ Please train the model from the sidebar to enable face recognition for this student.")

def mark_attendance_page():
    """Mark attendance page"""
    st.markdown('<h1 class="big-title">📸 Mark Attendance</h1>', unsafe_allow_html=True)
    
    if not is_model_trained():
        st.markdown("""
            <div class="error-box">
                <h3>⚠️ Model Not Trained</h3>
                <p>Please train the face recognition model first from the sidebar menu.</p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
        <div class="info-box">
            <p>📌 Click the button below to open your camera and mark attendance automatically.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("📷 Open Camera", use_container_width=True):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access camera. Please check your webcam connection.")
            return
        
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        result_placeholder = st.empty()
        
        stop_button = st.button("⏹️ Stop Camera")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(display_frame, channels="RGB", use_container_width=True)
            
            student_id, confidence = predict_face(frame)
            
            if student_id is not None:
                student = get_student_by_id(student_id)
                if student:
                    mark_attendance(student_id, student['name'])
                    cap.release()
                    
                    result_placeholder.markdown(f"""
                        <div class="success-box">
                            <h2>✅ Attendance Marked!</h2>
                            <p><strong>Name:</strong> {student['name']}</p>
                            <p><strong>Roll Number:</strong> {student['roll_number']}</p>
                            <p><strong>Class:</strong> {student['class']}</p>
                            <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
                            <p><strong>Time:</strong> {datetime.now().strftime('%I:%M %p')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    camera_placeholder.empty()
                    status_placeholder.empty()
                    break
            
            time.sleep(0.1)
        
        if cap.isOpened():
            cap.release()

def view_students_page():
    """View and manage students"""
    st.markdown('<h1 class="big-title">👥 Student Management</h1>', unsafe_allow_html=True)
    
    students = get_all_students()
    
    if not students:
        st.info("📝 No students registered yet. Register your first student from the sidebar!")
        return
    
    st.markdown(f"### Total Students: {len(students)}")
    
    search = st.text_input("🔍 Search by name or roll number", placeholder="Start typing...")
    
    if search:
        students = [s for s in students if search.lower() in s['name'].lower() or 
                   search.lower() in (s['roll_number'] or '').lower()]
    
    for student in students:
        with st.expander(f"👤 {student['name']} - {student['roll_number'] or 'N/A'}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Student ID:** {student['id']}")
                st.write(f"**Roll Number:** {student['roll_number'] or 'N/A'}")
                st.write(f"**Class:** {student['class'] or 'N/A'}")
                st.write(f"**Section:** {student['section'] or 'N/A'}")
                st.write(f"**Registration Number:** {student['registration_number'] or 'N/A'}")
                st.write(f"**Registered On:** {student['created_at'][:10]}")
            
            with col2:
                if st.button(f"🗑️ Delete", key=f"del_{student['id']}"):
                    delete_student(student['id'])
                    delete_student_images(student['id'])
                    st.success("✅ Student deleted!")
                    st.rerun()

def view_records_page():
    """View attendance records"""
    st.markdown('<h1 class="big-title">📋 Attendance Records</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        period = st.selectbox("📅 Filter by Period", 
                             ["All Time", "Today", "This Week", "This Month"])
    
    period_map = {
        "All Time": "all",
        "Today": "today",
        "This Week": "week",
        "This Month": "month"
    }
    
    df = get_attendance_records(period_map[period])
    
    if df.empty:
        st.info("📝 No attendance records found for the selected period.")
        return
    
    with col2:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown(f"### Total Records: {len(df)}")
    
    st.dataframe(
        df[['name', 'student_id', 'date', 'time']],
        use_container_width=True,
        height=400
    )
    
    if len(df) > 0:
        st.markdown("### 📊 Attendance Distribution")
        attendance_by_student = df.groupby('name').size().reset_index(name='count')
        attendance_by_student = attendance_by_student.sort_values('count', ascending=False).head(10)
        
        fig = px.bar(
            attendance_by_student,
            x='name',
            y='count',
            title='Top 10 Students by Attendance',
            color='count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def bulk_import_page():
    """Bulk import students from CSV"""
    st.markdown('<h1 class="big-title">📤 Bulk Import Students</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h3>📋 Import Multiple Students</h3>
            <p>Upload a CSV file to add multiple students at once. Download the template to see the required format.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📥 Upload CSV File")
        st.markdown("**Required columns:** name, roll_number, class, section, registration_number")
        st.markdown("**Note:** Only the 'name' column is required. Others are optional.")
    
    with col2:
        template_data = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'roll_number': ['2024001', '2024002', '2024003'],
            'class': ['10th Grade', '10th Grade', '9th Grade'],
            'section': ['A', 'B', 'A'],
            'registration_number': ['REG001', 'REG002', 'REG003']
        })
        
        csv_template = template_data.to_csv(index=False)
        st.download_button(
            label="📄 Download Template",
            data=csv_template,
            file_name="student_import_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### 📊 Preview Data")
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"Total rows in file: {len(df)}")
            
            required_cols = ['name']
            optional_cols = ['roll_number', 'class', 'section', 'registration_number']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return
            
            for col in optional_cols:
                if col not in df.columns:
                    df[col] = ''
            
            if st.button("🚀 Import Students", use_container_width=True, type="primary"):
                with st.spinner("Importing students..."):
                    students_data = df.to_dict('records')
                    success_count, errors = bulk_import_students(students_data)
                    
                    if success_count > 0:
                        st.markdown(f"""
                            <div class="success-box">
                                <h3>✅ Import Successful!</h3>
                                <p><strong>Successfully imported:</strong> {success_count} students</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    
                    if errors:
                        st.markdown("### ⚠️ Errors")
                        for error in errors:
                            st.warning(error)
                    
                    if success_count > 0:
                        st.info("💡 Reminder: Please train the model after importing students to enable face recognition.")
                        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please make sure your CSV file is properly formatted and uses comma (,) as delimiter.")

def advanced_analytics_page():
    """Advanced analytics and insights"""
    st.markdown('<h1 class="big-title">📊 Advanced Analytics</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "👥 Student Performance", "📚 Class Analysis"])
    
    with tab1:
        st.markdown("### 📈 Attendance Trends & Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            period = st.selectbox("Select Period", ["Last 7 Days", "Last 30 Days", "Last 90 Days"])
        
        with col2:
            chart_type = st.selectbox("Chart Type", ["Line Chart", "Bar Chart", "Area Chart"])
        
        days = 7 if period == "Last 7 Days" else 30 if period == "Last 30 Days" else 90
        dates, counts = get_attendance_stats(days)
        
        if chart_type == "Line Chart":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=counts,
                mode='lines+markers',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10, color='#764ba2'),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
        elif chart_type == "Bar Chart":
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dates, y=counts,
                marker=dict(color=counts, colorscale='Viridis')
            ))
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=counts,
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='#667eea', width=2)
            ))
        
        fig.update_layout(
            title=f"Attendance Trend - {period}",
            xaxis_title="Date",
            yaxis_title="Number of Students",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if counts:
            avg_attendance = sum(counts) / len(counts)
            max_attendance = max(counts)
            min_attendance = min(counts)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Attendance", f"{avg_attendance:.1f}")
            with col2:
                st.metric("Peak Attendance", max_attendance)
            with col3:
                st.metric("Lowest Attendance", min_attendance)
    
    with tab2:
        st.markdown("### 👥 Student Attendance Performance")
        
        summary_df = get_student_attendance_summary()
        
        if not summary_df.empty:
            st.dataframe(
                summary_df[['name', 'roll_number', 'class', 'section', 'total_attendance', 'last_attendance']],
                use_container_width=True,
                height=400
            )
            
            top_10 = summary_df.head(10)
            fig = px.bar(
                top_10,
                x='name',
                y='total_attendance',
                title='Top 10 Students by Attendance',
                color='total_attendance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No attendance data available yet.")
    
    with tab3:
        st.markdown("### 📚 Class-wise & Section-wise Analysis")
        
        period = st.radio("Select Period", ["Today", "This Week", "This Month", "All Time"], horizontal=True)
        period_map = {"Today": "today", "This Week": "week", "This Month": "month", "All Time": "all"}
        
        class_df = get_class_wise_attendance(period_map[period])
        
        if not class_df.empty and 'class' in class_df.columns:
            st.dataframe(class_df, use_container_width=True, height=300)
            
            fig = px.bar(
                class_df,
                x='class',
                y='attendance_rate',
                color='section',
                title=f'Class-wise Attendance Rate - {period}',
                labels={'attendance_rate': 'Attendance Rate (%)'},
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No class/section data available yet. Make sure students are assigned to classes and sections.")

def class_reports_page():
    """Class-wise and section-wise reports"""
    st.markdown('<h1 class="big-title">📑 Class & Section Reports</h1>', unsafe_allow_html=True)
    
    students = get_all_students()
    
    if not students:
        st.info("No students registered yet.")
        return
    
    classes = sorted(list(set([s['class'] for s in students if s['class']])))
    sections = sorted(list(set([s['section'] for s in students if s['section']])))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_class = st.selectbox("Select Class", ["All"] + classes)
    
    with col2:
        selected_section = st.selectbox("Select Section", ["All"] + sections)
    
    with col3:
        period = st.selectbox("Period", ["Today", "This Week", "This Month", "All Time"])
    
    period_map = {"Today": "today", "This Week": "week", "This Month": "month", "All Time": "all"}
    
    df = get_attendance_records(period_map[period])
    
    if not df.empty:
        student_df = pd.DataFrame(students)
        merged_df = df.merge(student_df[['id', 'class', 'section']], 
                             left_on='student_id', right_on='id', how='left')
        
        if selected_class != "All":
            merged_df = merged_df[merged_df['class'] == selected_class]
        
        if selected_section != "All":
            merged_df = merged_df[merged_df['section'] == selected_section]
        
        st.markdown(f"### 📊 Report Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            unique_students = merged_df['student_id'].nunique()
            st.metric("Unique Students", unique_students)
        
        with col2:
            total_records = len(merged_df)
            st.metric("Total Records", total_records)
        
        with col3:
            if not merged_df.empty:
                avg_per_student = total_records / unique_students if unique_students > 0 else 0
                st.metric("Avg per Student", f"{avg_per_student:.1f}")
        
        st.markdown("### 📋 Detailed Records")
        
        if not merged_df.empty:
            display_df = merged_df[['name', 'class', 'section', 'date', 'time']].copy()
            st.dataframe(display_df, use_container_width=True, height=400)
            
            csv_data = merged_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Report (CSV)",
                data=csv_data,
                file_name=f"class_report_{selected_class}_{selected_section}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No records found for the selected filters.")
    else:
        st.info("No attendance records available yet.")

def manual_attendance_page():
    """Manual attendance override and camera selection"""
    st.markdown('<h1 class="big-title">✍️ Manual Attendance</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["✍️ Manual Entry", "📷 Camera Settings"])
    
    with tab1:
        st.markdown("### ✍️ Manually Mark Attendance")
        st.info("Use this option when face recognition is not available or needs override.")
        
        students = get_all_students()
        
        if not students:
            st.warning("No students registered yet.")
            return
        
        student_options = {f"{s['name']} - {s['roll_number'] or 'N/A'}": s['id'] for s in students}
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_student = st.selectbox("Select Student", list(student_options.keys()))
        
        with col2:
            if st.button("✅ Mark Present", use_container_width=True, type="primary"):
                student_id = student_options[selected_student]
                student = get_student_by_id(student_id)
                
                if student:
                    mark_attendance(student_id, student['name'])
                    st.success(f"✅ Attendance marked for {student['name']}")
                    st.balloons()
        
        st.markdown("---")
        st.markdown("### 📋 Today's Manual Entries")
        
        today_df = get_attendance_records("today")
        if not today_df.empty:
            st.dataframe(
                today_df[['name', 'student_id', 'time']],
                use_container_width=True,
                height=300
            )
        else:
            st.info("No attendance records for today yet.")
    
    with tab2:
        st.markdown("### 📷 Camera Source Selection")
        
        st.info("Select which camera to use for face recognition attendance marking.")
        
        num_cameras = st.number_input("Number of available cameras", min_value=1, max_value=5, value=1)
        
        camera_index = st.selectbox("Select Camera", range(num_cameras), format_func=lambda x: f"Camera {x}")
        
        if st.button("🎥 Test Camera", use_container_width=True):
            cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    st.success(f"✅ Camera {camera_index} is working!")
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(display_frame, caption=f"Camera {camera_index} Preview", use_container_width=True)
                else:
                    st.error(f"❌ Could not read from Camera {camera_index}")
                cap.release()
            else:
                st.error(f"❌ Could not access Camera {camera_index}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Camera Settings")
        
        st.info("💡 Tip: The default camera (Camera 0) is usually the built-in webcam. External USB cameras are typically Camera 1, 2, etc.")
        
        if 'selected_camera' not in st.session_state:
            st.session_state.selected_camera = 0
        
        if st.button("💾 Save Camera Selection"):
            st.session_state.selected_camera = camera_index
            st.success(f"✅ Camera {camera_index} saved as default!")

def train_model_page():
    """Train model page"""
    st.markdown('<h1 class="big-title">🤖 Train Recognition Model</h1>', unsafe_allow_html=True)
    
    total_students = get_total_students()
    
    if total_students == 0:
        st.warning("⚠️ No students registered yet. Please register students first.")
        return
    
    st.markdown(f"""
        <div class="info-box">
            <h3>📚 Ready to Train</h3>
            <p><strong>Total Students:</strong> {total_students}</p>
            <p>Training will create a face recognition model that can identify registered students.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Start Training", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(progress / 100)
            status_text.text(message)
        
        success = train_model(update_progress)
        
        if success:
            st.markdown("""
                <div class="success-box">
                    <h2>✅ Training Completed Successfully!</h2>
                    <p>The face recognition model is now ready to use.</p>
                    <p>You can now mark attendance using the camera.</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.error("❌ Training failed. Please ensure students have face photos.")

def main():
    """Main application"""
    init_database()
    apply_custom_css()
    
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    with st.sidebar:
        st.markdown("## 🎓 Navigation")
        
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        
        st.markdown("### 📝 Student Management")
        if st.button("➕ Register Student", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()
        
        if st.button("📤 Bulk Import", use_container_width=True):
            st.session_state.page = 'bulk_import'
            st.rerun()
        
        if st.button("👥 Manage Students", use_container_width=True):
            st.session_state.page = 'students'
            st.rerun()
        
        st.markdown("### 📸 Attendance")
        if st.button("📸 Mark Attendance", use_container_width=True):
            st.session_state.page = 'mark_attendance'
            st.rerun()
        
        if st.button("✍️ Manual Entry", use_container_width=True):
            st.session_state.page = 'manual'
            st.rerun()
        
        st.markdown("### 📊 Reports & Analytics")
        if st.button("📋 View Records", use_container_width=True):
            st.session_state.page = 'records'
            st.rerun()
        
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.page = 'analytics'
            st.rerun()
        
        if st.button("📑 Class Reports", use_container_width=True):
            st.session_state.page = 'class_reports'
            st.rerun()
        
        st.markdown("### 🤖 System")
        if st.button("🤖 Train Model", use_container_width=True):
            st.session_state.page = 'train'
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("AI-Powered Face Recognition Attendance System for educational institutions.")
        
        model_status = "✅ Trained" if is_model_trained() else "❌ Not Trained"
        st.markdown(f"**Model Status:** {model_status}")
        st.markdown(f"**Students:** {get_total_students()}")
    
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'register':
        register_student_page()
    elif st.session_state.page == 'bulk_import':
        bulk_import_page()
    elif st.session_state.page == 'mark_attendance':
        mark_attendance_page()
    elif st.session_state.page == 'manual':
        manual_attendance_page()
    elif st.session_state.page == 'students':
        view_students_page()
    elif st.session_state.page == 'records':
        view_records_page()
    elif st.session_state.page == 'analytics':
        advanced_analytics_page()
    elif st.session_state.page == 'class_reports':
        class_reports_page()
    elif st.session_state.page == 'train':
        train_model_page()

if __name__ == "__main__":
    main()

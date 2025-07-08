import sys
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFormLayout, QGroupBox, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

# --- Video Processing and Biomechanics Thread ---
class VideoThread(QThread):
    # Signals to emit processed frame and calculated data
    change_pixmap_signal = pyqtSignal(QPixmap)
    update_data_signal = pyqtSignal(dict)
    camera_error_signal = pyqtSignal(str) # Signal for camera errors
    
    def __init__(self, constants):
        super().__init__()
        self._run_flag = True
        self.constants = constants
        self.log_data = [] # To store data for CSV export
        # Store headers for CSV/graphing purposes
        self.log_headers = [
            "Time", "Knee Angle (deg)", "Hip Angle (deg)", "Back Angle (deg)",
            "Quad Force (N)", "Glute Force (N)", "Hamstring Force (N)", "Back Force (N)",
            "Strain", "Quad Young Modulus (Pa)", "Glute Young Modulus (Pa)",
            "Hamstring Young Modulus (Pa)", "Back Young Modulus (Pa)"
        ]

    def run(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        self.log_data = [] # Clear log data on new run
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.camera_error_signal.emit("Could not open webcam. Please ensure it's connected and not in use by another application.")
            self._run_flag = False
            return

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, attempting to re-open camera...")
                cap.release()
                time.sleep(1) # Give a moment before trying again
                cap = cv2.VideoCapture(0) # Attempt to re-open
                if not cap.isOpened():
                    self.camera_error_signal.emit("Could not re-open webcam. Analysis stopped.")
                    self._run_flag = False
                    break
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            data_to_display = {
                "Knee Angle": 0, "Hip Angle": 0, "Back Angle": 0,
                "Quad F": 0, "Glute F": 0, "Hamstring F": 0, "Back F": 0,
                "Strain": 0,
                "Quad E": 0, "Glute E": 0, "Hamstring E": 0, "Back E": 0
            }

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                try:
                    shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
                    hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])
                    knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y])
                    ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y])
                except Exception as e:
                    print(f"Partial landmark detection or error: {e}")
                    self._draw_and_emit_frame(image, results, mp_pose)
                    self.update_data_signal.emit(data_to_display)
                    continue

                knee_angle = self._calculate_angle(hip, knee, ankle)
                hip_angle = self._calculate_angle(shoulder, hip, knee)
                back_angle = self._calculate_angle(shoulder, hip, knee)

                theta_knee = math.radians(180 - knee_angle)
                theta_hip = math.radians(180 - hip_angle)

                F_external = (self.constants['weight_kg'] * 9.81) / 2 
                phi_rad = self.constants['phi']
                F_parallel = F_external * math.cos(phi_rad)

                tau_knee = F_parallel * self.constants['r_external']
                tau_hip = F_parallel * self.constants['r_external']

                F_quad = self._calculate_force(tau_knee, self.constants['r_quad'], theta_knee)
                F_glute = self._calculate_force(tau_hip, self.constants['r_glute'], theta_hip)
                F_hamstring = self._calculate_force(tau_hip, self.constants['r_hamstring_hip'], theta_hip)
                F_erector_spinae = self._calculate_force(tau_hip, self.constants['r_erector_spinae'], theta_hip)

                # Strain calculation using delta_L_ratio
                delta_L = self.constants['L0'] * self.constants['delta_L_ratio']
                strain = delta_L / self.constants['L0'] if self.constants['L0'] != 0 else 0

                E_quad = self._calculate_young_modulus(F_quad, self.constants['L0'], self.constants['A_quad'], delta_L)
                E_glute = self._calculate_young_modulus(F_glute, self.constants['L0'], self.constants['A_glute'], delta_L)
                E_hamstring = self._calculate_young_modulus(F_hamstring, self.constants['L0'], self.constants['A_hamstring'], delta_L)
                E_erector_spinae = self._calculate_young_modulus(F_erector_spinae, self.constants['L0'], self.constants['A_erector_spinae'], delta_L)

                data_to_display = {
                    "Knee Angle": int(knee_angle),
                    "Hip Angle": int(hip_angle),
                    "Back Angle": int(back_angle),
                    "Quad F": int(F_quad),
                    "Glute F": int(F_glute),
                    "Hamstring F": int(F_hamstring),
                    "Back F": int(F_erector_spinae),
                    "Strain": strain * 100, # Convert to percentage
                    "Quad E": E_quad / 1e6, # Convert to MPa
                    "Glute E": E_glute / 1e6, # Convert to MPa
                    "Hamstring E": E_hamstring / 1e6, # Convert to MPa
                    "Back E": E_erector_spinae / 1e6 # Convert to MPa
                }

                timestamp = time.time()
                self.log_data.append([
                    timestamp, knee_angle, hip_angle, back_angle,
                    F_quad, F_glute, F_hamstring, F_erector_spinae,
                    strain, E_quad, E_glute, E_hamstring, E_erector_spinae
                ])

            self._draw_and_emit_frame(image, results, mp_pose)
            self.update_data_signal.emit(data_to_display)

        cap.release()
        pose.close()
        print("Video thread stopped.")

    def _calculate_angle(self, a, b, c):
        ab = a - b
        cb = c - b
        cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _calculate_force(self, tau, r, theta):
        # Prevent division by zero or near-zero cosine values which can lead to extreme force estimates
        if r == 0 or math.isclose(math.cos(theta), 0, abs_tol=1e-9):
            return 0
        return tau / (r * math.cos(theta))

    def _calculate_young_modulus(self, F, L0, A, delta_L):
        # Prevent division by zero
        if A * delta_L == 0:
            return 0
        return (F * L0) / (A * delta_L)

    def _draw_and_emit_frame(self, image, results, mp_pose_local):
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                      mp_pose_local.POSE_CONNECTIONS,
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                     )
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.change_pixmap_signal.emit(QPixmap.fromImage(p))

    def stop(self):
        self._run_flag = False
        self.wait()

# --- Main GUI Application ---
class BiomechanicsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Squat Biomechanics Analysis")
        self.setGeometry(100, 100, 1200, 700)

        self.constants = self._get_default_constants()
        self.video_thread = None

        self._init_ui()
        self._set_initial_focus() # Set initial focus after UI is built

        # Apply stylesheet
        self.setStyleSheet(self._get_stylesheet())

    def _get_default_constants(self):
        return {
            'A_quad': 0.00006,
            'A_glute': 0.00009,
            'A_hamstring': 0.00007,
            'A_erector_spinae': 0.00003,
            'L0': 0.10,
            'r_quad': 0.04,
            'r_glute': 0.07,
            'r_hamstring_hip': 0.06,
            'r_erector_spinae': 0.05,
            'r_external': 0.35,
            'weight_kg': 100.0,
            'phi_deg': 15.0,
            'phi': math.radians(15.0), # Calculated from phi_deg
            'delta_L_ratio': 0.08,
            'age_years': 30 # Default age
        }

    def _init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20) # Add margin around the main layout
        main_layout.setSpacing(20) # Spacing between left panel and video feed

        # Left Panel: Controls and Data Display
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setSpacing(15) # Spacing between groups

        # Basic Constants Input Group
        basic_constants_group = QGroupBox("Basic Constants")
        basic_constants_form_layout = QFormLayout()
        basic_constants_form_layout.setContentsMargins(10, 15, 10, 10) # Padding inside group box
        basic_constants_form_layout.setSpacing(8) # Spacing between form rows

        self.constant_inputs = {} # Store all inputs, even advanced ones
        
        basic_constant_definitions = [
            ("Weight (kg):", 'weight_kg', "Enter the total weight being lifted (body weight + barbell weight)."),
            ("External Moment Arm (m):", 'r_external', "Distance from the joint (hip/knee) to the line of action of the external force. Adjust based on barbell position."),
            ("Hip Abduction Angle (deg):", 'phi_deg', "Angle representing the load's line of action relative to the body's sagital plane. Helps account for non-vertical force components."),
            ("Quad Moment Arm (m):", 'r_quad', "Effective moment arm of the quadriceps muscle around the knee joint. A key factor in quadriceps force calculation."),
            ("Glute Moment Arm (m):", 'r_glute', "Effective moment arm of the gluteal muscles around the hip joint. Important for hip extension force."),
            ("Hamstring Moment Arm (m):", 'r_hamstring_hip', "Effective moment arm of the hamstrings around the hip joint (primarily for hip extension in squat)."),
            ("Erector Spinae Moment Arm (m):", 'r_erector_spinae', "Effective moment arm of the erector spinae (back muscles) around the hip joint, crucial for maintaining torso uprightness.")
        ]

        for label_text, key, tooltip_text in basic_constant_definitions:
            line_edit = QLineEdit(str(self.constants[key]))
            line_edit.setValidator(QDoubleValidator())
            line_edit.setToolTip(tooltip_text) # Add tooltip
            basic_constants_form_layout.addRow(label_text, line_edit)
            self.constant_inputs[key] = line_edit
        
        basic_constants_group.setLayout(basic_constants_form_layout)
        left_panel_layout.addWidget(basic_constants_group)

        # Advanced Constants Input Group (Collapsible)
        self.advanced_constants_group = QGroupBox("Advanced Anatomical Constants")
        self.advanced_constants_group.setCheckable(True) # Make it checkable to act as a toggle
        self.advanced_constants_group.toggled.connect(self._toggle_advanced_constants)
        
        advanced_constants_form_layout = QFormLayout()
        advanced_constants_form_layout.setContentsMargins(10, 15, 10, 10)
        advanced_constants_form_layout.setSpacing(8)

        advanced_constant_definitions = [
            ("Age (years):", 'age_years', "Your age in years. Used to estimate a general physiological strain limit for tendons."),
            ("Quad Area (m²):", 'A_quad', "Physiological cross-sectional area of the quadriceps muscle. Larger area generally means higher force potential."),
            ("Glute Area (m²):", 'A_glute', "Physiological cross-sectional area of the gluteal muscles. Affects glute force and Young's Modulus."),
            ("Hamstring Area (m²):", 'A_hamstring', "Physiological cross-sectional area of the hamstring muscles. Influences hamstring force contribution."),
            ("Erector Spinae Area (m²):", 'A_erector_spinae', "Physiological cross-sectional area of the erector spinae muscles. Critical for back extensor force."),
            ("Resting Tendon Length (m):", 'L0', "Initial, unstretched length of the muscle-tendon unit. Baseline for strain calculation."),
            ("Delta L Ratio:", 'delta_L_ratio', "Ratio of the change in tendon length to the resting length. Directly influences the calculated strain.")
        ]

        for label_text, key, tooltip_text in advanced_constant_definitions:
            line_edit = QLineEdit(str(self.constants[key]))
            if key == 'age_years':
                line_edit.setValidator(QDoubleValidator(0, 150, 0)) # Age usually int, but QDoubleValidator can take 0 decimals
            else:
                line_edit.setValidator(QDoubleValidator())
            line_edit.setToolTip(tooltip_text) # Add tooltip
            advanced_constants_form_layout.addRow(label_text, line_edit)
            self.constant_inputs[key] = line_edit # Add to the same dictionary
            
        self.advanced_constants_group.setLayout(advanced_constants_form_layout)
        left_panel_layout.addWidget(self.advanced_constants_group)
        
        # Control Buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 10) # Padding around buttons
        button_layout.setSpacing(10) # Spacing between buttons

        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Analysis")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        left_panel_layout.addLayout(button_layout)

        # Save Options
        save_button_layout = QHBoxLayout()
        save_button_layout.setContentsMargins(0, 0, 0, 10)
        save_button_layout.setSpacing(10)

        self.save_csv_button = QPushButton("Save Data to CSV")
        self.save_csv_button.clicked.connect(self.save_data_to_csv)
        self.save_csv_button.setEnabled(False)
        save_button_layout.addWidget(self.save_csv_button)

        self.save_graph_button = QPushButton("Save Data as Graph")
        self.save_graph_button.clicked.connect(self.save_data_as_graph)
        self.save_graph_button.setEnabled(False)
        save_button_layout.addWidget(self.save_graph_button)
        
        left_panel_layout.addLayout(save_button_layout)

        # Real-time Data Display Group
        data_group = QGroupBox("Real-time Biomechanics Data")
        data_form_layout = QFormLayout()
        data_form_layout.setContentsMargins(10, 15, 10, 10)
        data_form_layout.setSpacing(8)

        self.data_labels = {}

        data_fields = [
            "Knee Angle (deg):", "Hip Angle (deg):", "Back Angle (deg):",
            "Quad Force (N):", "Glute Force (N):", "Hamstring Force (N):", "Back Force (N):",
            "Strain (%):",
            "Quad Young Modulus (MPa):", "Glute Young Modulus (MPa):",
            "Hamstring Young Modulus (MPa):", "Back Young Modulus (MPa):"
        ]

        for field in data_fields:
            label = QLabel("N/A") # Initialize with "N/A"
            label.setProperty("dataLabel", True) # Set custom property for QSS
            data_form_layout.addRow(field, label)
            self.data_labels[field.split('(')[0].strip()] = label
        
        data_group.setLayout(data_form_layout)
        left_panel_layout.addWidget(data_group)

        left_panel_layout.addStretch() # Pushes content to the top
        main_layout.addLayout(left_panel_layout)

        # Right Panel: Video Feed
        self.image_label = QLabel("Click 'Start Analysis' to begin live camera feed.")
        self.image_label.setObjectName("image_label") # Set object name for QSS
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480) # Fixed size for video feed
        main_layout.addWidget(self.image_label)

        # Set the main layout for the widget
        self.setLayout(main_layout)

        # After the layout is set for the main widget, then trigger the collapse
        # This ensures self.layout() is not None when _toggle_advanced_constants is called
        self.advanced_constants_group.setChecked(False) # Start collapsed

    def _set_initial_focus(self):
        # Set focus to the first basic constant input
        if 'weight_kg' in self.constant_inputs: # Reverted to weight_kg as first basic constant
            self.constant_inputs['weight_kg'].setFocus()

    def _toggle_advanced_constants(self, checked):
        # Toggle visibility of all widgets within the advanced_constants_group's layout
        for i in range(self.advanced_constants_group.layout().count()):
            widget = self.advanced_constants_group.layout().itemAt(i).widget()
            if widget:
                widget.setVisible(checked)
        
        # Adjust size policy for collapsing effect
        if not checked:
            self.advanced_constants_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.advanced_constants_group.layout().setContentsMargins(10, 0, 10, 0) # Reduce top/bottom margin when collapsed
        else:
            self.advanced_constants_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            self.advanced_constants_group.layout().setContentsMargins(10, 15, 10, 10) # Restore padding when expanded
        # Update layout to reflect changes
        self.advanced_constants_group.adjustSize()
        # Invalidate the parent layout to force recalculation
        if self.layout(): # Check if layout exists before invalidating
            self.layout().invalidate()

    def _estimate_max_physiological_strain(self, age):
        """
        Estimates a general maximum physiological strain percentage for tendons based on age.
        This is a simplified model for guidance and is not a precise medical limit.
        """
        if age < 0: return 0 # Invalid age
        if age < 20: return 5.0 # Young, more elastic
        elif age < 40: return 4.0 # Adult, good elasticity
        elif age < 60: return 3.0 # Middle-aged, decreased elasticity
        else: return 2.0 # Older, less elastic
        
    def _update_constants_from_gui(self):
        try:
            for key, line_edit in self.constant_inputs.items():
                if key == 'phi_deg':
                    self.constants[key] = float(line_edit.text())
                    self.constants['phi'] = math.radians(self.constants[key])
                elif key == 'age_years':
                    # Ensure age is an integer for clarity in the model, even if validator is float
                    self.constants[key] = int(float(line_edit.text()))
                else:
                    self.constants[key] = float(line_edit.text())
            return True
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numerical values for all constants.")
            return False

    @pyqtSlot(QPixmap)
    def set_image(self, pixmap):
        self.image_label.setPixmap(pixmap)
        self.image_label.setText("") # Clear any status message when image is displayed

    @pyqtSlot(dict)
    def update_labels(self, data):
        for key_display, label_widget in self.data_labels.items():
            if key_display in data:
                value = data[key_display]
                if "Force" in key_display or "Angle" in key_display:
                    label_widget.setText(f"{value}")
                elif "Strain" in key_display:
                    label_widget.setText(f"{value:.2f}")
                elif "Young Modulus" in key_display:
                    label_widget.setText(f"{value:.2f}")
            else:
                label_widget.setText("N/A")

    @pyqtSlot(str)
    def _show_camera_error(self, message):
        self.image_label.setText(f"CAMERA ERROR\n\n{message}") # Display error directly on video label
        QMessageBox.critical(self, "Camera Error", message)
        self.stop_analysis() # Stop analysis immediately on camera error

    def start_analysis(self):
        if not self._update_constants_from_gui():
            return

        for label_widget in self.data_labels.values():
            label_widget.setText("0") # Reset data labels to 0 when starting

        # Initial check for camera access directly in GUI thread to give immediate feedback
        temp_cap = cv2.VideoCapture(0)
        if not temp_cap.isOpened():
            self._show_camera_error("Could not open webcam. Please ensure it's connected and not in use by another application.")
            temp_cap.release()
            return
        temp_cap.release() # Release it quickly

        if self.video_thread is not None and self.video_thread.isRunning():
            QMessageBox.information(self, "Analysis Running", "Analysis is already running.")
            return

        self.image_label.setText("Starting analysis...\nPlease ensure good lighting and clear view of your right side.") # Status message
        self.video_thread = VideoThread(self.constants)
        self.video_thread.change_pixmap_signal.connect(self.set_image)
        self.video_thread.update_data_signal.connect(self.update_labels)
        self.video_thread.camera_error_signal.connect(self._show_camera_error) # Connect new signal
        self.video_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_csv_button.setEnabled(True)
        self.save_graph_button.setEnabled(True)
        for line_edit in self.constant_inputs.values():
            line_edit.setEnabled(False)
        self.advanced_constants_group.setEnabled(False) # Disable toggling advanced constants


    def stop_analysis(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            pass 
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        for line_edit in self.constant_inputs.values():
            line_edit.setEnabled(True)
        self.advanced_constants_group.setEnabled(True) # Re-enable toggling
        self.image_label.setText("Analysis stopped.\nClick 'Start Analysis' to begin live camera feed.") # Status message


    def save_data_to_csv(self):
        if self.video_thread is None or not self.video_thread.log_data:
            QMessageBox.information(self, "No Data", "No data has been collected yet to save.")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Save Biomechanics Data", "squat_biomechanics_log.csv", "CSV Files (*.csv)")
        if file_name:
            try:
                with open(file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.video_thread.log_headers)
                    writer.writerows(self.video_thread.log_data)
                QMessageBox.information(self, "Save Successful", f"Data saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save data: {e}")

    def save_data_as_graph(self):
        if self.video_thread is None or not self.video_thread.log_data:
            QMessageBox.information(self, "No Data", "No data has been collected yet to plot.")
            return

        if self.video_thread.isRunning():
            reply = QMessageBox.question(self, 'Stop Analysis?', 
                                         "To save a complete graph, the analysis should be stopped. Stop now?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_analysis()
            else:
                return
        
        # Ensure constants are updated for the graph (especially age)
        if not self._update_constants_from_gui():
            return


        log_data = np.array(self.video_thread.log_data)
        if log_data.shape[0] < 2:
            QMessageBox.information(self, "Not Enough Data", "Not enough data points collected to create a meaningful graph.")
            return

        timestamps_raw = log_data[:, 0]
        time_elapsed = timestamps_raw - timestamps_raw[0]

        headers = self.video_thread.log_headers
        
        angles_indices = [1, 2, 3]
        forces_indices = [4, 5, 6, 7]
        moduli_indices = [9, 10, 11, 12]
        strain_index = 8

        try:
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 16), sharex=True)
            fig.suptitle('Squat Biomechanics Analysis Over Time', fontsize=16)

            ax0 = axes[0]
            for i in angles_indices:
                ax0.plot(time_elapsed, log_data[:, i], label=headers[i].split('(')[0].strip())
            ax0.set_ylabel('Angle (deg)')
            ax0.legend()
            ax0.grid(True)
            ax0.set_title('Joint Angles')

            ax1 = axes[1]
            for i in forces_indices:
                ax1.plot(time_elapsed, log_data[:, i], label=headers[i].split('(')[0].strip())
            ax1.set_ylabel('Force (N)')
            ax1.legend()
            ax1.grid(True)
            ax1.set_title('Muscle Forces')

            ax2 = axes[2]
            ax2.plot(time_elapsed, log_data[:, strain_index] * 100, label=headers[strain_index].split('(')[0].strip(), color='purple')
            ax2.set_ylabel('Strain (%)')
            ax2.legend()
            ax2.grid(True)
            ax2.set_title('Strain')

            # Add horizontal line for estimated physiological strain limit
            age = self.constants.get('age_years', 30) # Get age from constants, default to 30 if not found
            max_strain_limit = self._estimate_max_physiological_strain(age)
            ax2.axhline(y=max_strain_limit, color='red', linestyle='--', label=f'Estimated Max Strain for Age {age}: {max_strain_limit:.1f}%')
            ax2.legend() # Update legend to include the new line

            ax3 = axes[3]
            for i in moduli_indices:
                ax3.plot(time_elapsed, log_data[:, i] / 1e6, label=headers[i].split('(')[0].strip())
            ax3.set_ylabel('Young\'s Modulus (MPa)')
            ax3.legend()
            ax3.grid(True)
            ax3.set_title('Young\'s Moduli')
            ax3.set_xlabel('Time (seconds)')

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])

            file_name, _ = QFileDialog.getSaveFileName(self, "Save Biomechanics Graph", "squat_biomechanics_graph.png", "PNG Image (*.png);;PDF Document (*.pdf);;All Files (*)")
            if file_name:
                fig.savefig(file_name)
                QMessageBox.information(self, "Save Successful", f"Graph saved to {file_name}")
            else:
                QMessageBox.information(self, "Save Cancelled", "Graph saving cancelled.")

            plt.close(fig)
        
        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"Failed to generate or save graph: {e}")
            print(f"Plotting error: {e}")

    def closeEvent(self, event):
        if self.video_thread is not None:
            self.video_thread.stop()
        event.accept()

    def _get_stylesheet(self):
        return """
            QWidget {
                background-color: #f0f2f5; /* Light grey-blue background */
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                color: #333333;
            }

            QGroupBox {
                background-color: #ffffff; /* White background for groups */
                border: 1px solid #d8e0e7; /* Light grey border */
                border-radius: 10px; /* Slightly more rounded corners */
                margin-top: 15px; /* Space for title */
                padding-top: 25px; /* Adjust internal padding for title space */
                font-weight: bold;
                color: #2c3e50; /* Darker text for titles */
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 3px 10px;
                background-color: #e9ecef; /* Slightly darker title background */
                border-radius: 7px;
                margin-left: 10px;
                font-size: 15px; /* Slightly larger title font */
            }

            QLineEdit {
                border: 1px solid #ced4da; /* Bootstrap-like border color */
                border-radius: 5px;
                padding: 6px 8px; /* More padding */
                background-color: #f8f9fa;
                color: #333333;
            }

            QPushButton {
                background-color: #007bff; /* Primary blue */
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 18px;
                margin: 5px 0; /* Add vertical margin */
                font-weight: bold;
                min-width: 80px; /* Ensure minimum width */
            }

            QPushButton:hover {
                background-color: #0056b3; /* Darker blue on hover */
            }

            QPushButton:pressed {
                background-color: #004085; /* Even darker on press */
            }

            QPushButton:disabled {
                background-color: #e9ecef;
                color: #adb5bd;
            }

            QLabel {
                color: #495057; /* Darker grey for general labels */
            }

            /* Styling for real-time data labels specifically */
            QLabel[property="dataLabel"] { 
                font-size: 19px; /* Larger font */
                font-weight: bold;
                color: #28a745; /* Green for data */
                padding: 2px;
                background-color: transparent;
                min-width: 60px; /* Ensure some width for values */
            }

            #image_label { /* Targeting by object name */
                background-color: #343a40; /* Darker grey for video background */
                color: #f8f9fa; /* Light text color */
                border: 3px solid #6c757d; /* Medium grey border */
                border-radius: 12px; /* More rounded corners */
                font-size: 16px;
                font-weight: bold;
                padding: 20px;
            }
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BiomechanicsApp()
    window.show()
    sys.exit(app.exec_())

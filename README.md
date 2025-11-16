# AI-Powered Dual-Radar System for Real-Time Survivor Detection (Ongoing)

This is an ongoing B.Tech project to design and develop a portable system for real-time survivor detection in disaster zones. The primary objective is to detect human vital signs (respiration and heartbeat) through rubble.

### Project Architecture & Process
The system's architecture is built on the following process:

1.  **Sensor Fusion:** It utilizes a dual-radar architecture, combining FMCW mmWave and UWB radar to capture robust data.

2.  **Data & Image Processing:** Raw radar data is processed and converted into image-like representations (e.g., spectrograms). These images then undergo **image cleaning**, **image processing**, and **data handling** to remove noise and prepare the features for classification.

3.  **AI Classification (Current Stage):** We are currently using **SqueezeNet**, a lightweight and efficient Convolutional Neural Network (CNN), to **classify** the processed images and detect the presence of human vital signs.

4.  **Hardware Acceleration (Planned):** Signal processing and AI inference are being designed for implementation on an FPGA to ensure the system can perform real-time detection.

5.  **Secure Transmission (Planned):** All survivor alerts will be integrated with Post-Quantum Cryptography (PQC) to ensure data integrity and security.

### Technologies Used
* **Programming:** Python, MATLAB
* **AI/ML:** Deep Learning, **SqueezeNet (CNN)**, **Image Processing**, Data Handling, Pandas, Scikit-learn, (TensorFlow/PyTorch)
* **Hardware:** UWB Radar, FMCW mmWave Radar, (Planned: FPGA)
* **Security:** (Planned: Post-Quantum Cryptography (PQC) libraries)

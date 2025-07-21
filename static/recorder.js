// Đợi cho toàn bộ trang web được tải xong rồi mới chạy code
document.addEventListener('DOMContentLoaded', () => {
    // Lấy các phần tử HTML mà chúng ta cần tương tác
    const recordBtn = document.getElementById('record-btn');
    const timerElement = document.getElementById('timer');
    const resultsPanel = document.querySelector('.results-content');

    // Khai báo các biến cần thiết
    let mediaRecorder; // Đối tượng chính để ghi âm
    let audioChunks = []; // Mảng để lưu các mẩu âm thanh
    let timerInterval; // Biến để điều khiển đồng hồ đếm giờ
    let seconds = 0; // Số giây đã ghi âm

    // --- CÁC HÀM XỬ LÝ ---

    // Hàm chính để bắt đầu/dừng ghi âm
    const toggleRecording = async () => {
        // Trường hợp 1: Bắt đầu ghi âm
        if (!mediaRecorder || mediaRecorder.state === 'inactive') {
            try {
                // Yêu cầu quyền truy cập microphone
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                // Khởi tạo MediaRecorder với stream âm thanh
                mediaRecorder = new MediaRecorder(stream);

                // Sự kiện này được gọi khi có một mẩu dữ liệu âm thanh
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                // Sự kiện này được gọi khi quá trình ghi âm dừng lại
                mediaRecorder.onstop = () => {
                    // Tạo một file âm thanh hoàn chỉnh từ các mẩu đã thu
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    
                    // Tạo một URL tạm thời cho file âm thanh để có thể phát lại
                    const audioUrl = URL.createObjectURL(audioBlob);

                    // Hiển thị trình phát âm thanh để người dùng nghe lại
                    const audioPlayer = new Audio(audioUrl);
                    audioPlayer.controls = true; // Hiển thị các nút play/pause/volume
                    
                    // Xóa nội dung cũ và hiển thị trình phát âm thanh mới
                    resultsPanel.innerHTML = ''; // Xóa chữ "Kết quả sẽ hiển thị ở đây"
                    resultsPanel.appendChild(audioPlayer);
                    resultsPanel.insertAdjacentHTML('beforeend', '<p>Bản ghi của bạn đã sẵn sàng. Bạn có thể nghe lại.</p>');

                    // Reset mảng chứa các mẩu âm thanh cho lần ghi tiếp theo
                    audioChunks = [];
                };

                // Bắt đầu ghi âm
                mediaRecorder.start();
                updateButtonState('recording');
                startTimer();

            } catch (error) {
                // Xử lý lỗi nếu người dùng không cấp quyền
                console.error('Lỗi khi truy cập microphone:', error);
                resultsPanel.innerHTML = '<p style="color: red;">Không thể truy cập microphone. Vui lòng cấp quyền trong cài đặt trình duyệt và thử lại.</p>';
            }
        } else { // Trường hợp 2: Dừng ghi âm
            mediaRecorder.stop();
            updateButtonState('inactive');
            stopTimer();
        }
    };

    // Hàm cập nhật giao diện của nút bấm (màu sắc, icon, chữ)
    const updateButtonState = (state) => {
        const icon = recordBtn.querySelector('.icon');
        const text = recordBtn.querySelector('.text');

        if (state === 'recording') {
            recordBtn.style.backgroundColor = '#dc3545'; // Chuyển sang màu đỏ
            icon.textContent = '■'; // Biểu tượng stop
            text.textContent = 'DỪNG';
        } else {
            recordBtn.style.backgroundColor = '#28a745'; // Trở về màu xanh
            icon.textContent = '🎤'; // Biểu tượng micro
            text.textContent = 'GHI ÂM';
        }
    };

    // Hàm bắt đầu đồng hồ đếm giờ
    const startTimer = () => {
        seconds = 0;
        timerElement.textContent = '00:00';
        timerInterval = setInterval(() => {
            seconds++;
            const minutes = Math.floor(seconds / 60).toString().padStart(2, '0');
            const secs = (seconds % 60).toString().padStart(2, '0');
            timerElement.textContent = `${minutes}:${secs}`;
        }, 1000);
    };

    // Hàm dừng đồng hồ
    const stopTimer = () => {
        clearInterval(timerInterval);
    };

    // --- GÁN SỰ KIỆN ---

    // Gán sự kiện 'click' cho nút ghi âm để gọi hàm toggleRecording
    recordBtn.addEventListener('click', toggleRecording);
});
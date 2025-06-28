const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

// Start the webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

// Function to capture frame & send to FastAPI
async function captureAndSendFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert frame to Blob
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        // Send frame to FastAPI
        try {
            const response = await fetch("http://127.0.0.1:8000/predict/", {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            displayPredictions(result);
        } catch (error) {
            console.error("Error sending frame:", error);
        }
    }, "image/jpeg");
}

// Function to display predictions
function displayPredictions(data) {
    overlay.innerHTML = `
        <p><strong>Height:</strong> ${data.height} ft</p>
        <p><strong>Weight:</strong> ${data.weight} kg</p>
        <p><strong>Age:</strong> ${data.age}</p>
        <p><strong>Gender:</strong> ${data.gender}</p>
    `;
}

// Send a frame every second
setInterval(captureAndSendFrame, 1000);

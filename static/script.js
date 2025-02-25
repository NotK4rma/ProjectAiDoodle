const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear");
const submitButton = document.getElementById("submit");
const resultText = document.getElementById("result");

// Setup canvas for drawing
let drawing = false;

canvas.addEventListener("mousedown", () => (drawing = true));
canvas.addEventListener("mouseup", () => (drawing = false));
canvas.addEventListener("mouseout", () => (drawing = false));

canvas.addEventListener("mousemove", (event) => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  ctx.fillStyle = "black";
  ctx.fillRect(x, y, 10, 10); // Draw a small rectangle
});

// Clear the canvas
clearButton.addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  resultText.textContent = "";
});

// Submit the doodle
submitButton.addEventListener("click", async () => {
  const dataURL = canvas.toDataURL("image/png");
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataURL.split(",")[1] }), // Send only the base64 image
  });

  const result = await response.json();
  if (result.prediction) {
    resultText.textContent = `AI thinks you drew: ${result.prediction}`;
  } else {
    resultText.textContent = "Error: Could not get prediction.";
  }
});

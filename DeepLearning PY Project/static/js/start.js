document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault(); // Prevent default form submission

    const formData = new FormData(e.target);

    try {
        // Upload the image
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            alert("Error processing the image.");
            return;
        }

        const data = await response.json();

        // Display the images (original and processed)
        document.getElementById("imageDisplay").style.display = "block";
        document.getElementById("uploadedImage").src = data.original_image_url;
        document.getElementById("processedImage").src = data.processed_image_url;

        // Show the Predict button
        document.getElementById("predictSection").style.display = "block";

        // Add Predict button functionality
        document.getElementById("predictButton").addEventListener("click", async () => {
            // Disable the button after click to avoid multiple predictions
            document.getElementById("predictButton").disabled = true;
            document.getElementById("predictButton").textContent = "Predicting...";

            try {
                // Send the image to predict endpoint
                const predictResponse = await fetch("/predict", {
                    method: "POST",
                    body: formData,
                });

                if (!predictResponse.ok) {
                    alert("Error predicting the result.");
                    return;
                }

                const predictData = await predictResponse.json();

                // Display the prediction result
                document.getElementById("predictionResult").style.display = "block";
                document.getElementById("resultText").textContent = `Predicted Class: ${predictData.predicted_class}`;

                // Enable the button again
                document.getElementById("predictButton").disabled = false;
                document.getElementById("predictButton").textContent = "Predict";
            } catch (error) {
                alert("An error occurred during prediction.");
                document.getElementById("predictButton").disabled = false;
                document.getElementById("predictButton").textContent = "Predict";
            }
        });
    } catch (error) {
        alert("An error occurred during image upload.");
    }
});

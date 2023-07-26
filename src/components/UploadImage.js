import React, { useState } from 'react';
import axios from 'axios';
import './Uploadimage.css'; // Import the CSS file for styling

const UploadImage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [prediction, setPrediction] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);

    // Create a preview URL for the selected image
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handlePredict = () => {
    const formData = new FormData();
    formData.append('file', selectedFile);

    axios.post('http://localhost:5000/predict', formData)
      .then(response => {
        // Assuming the response data is in JSON format
        const predictionValue = response.data.prediction;

        // Update the prediction state with the extracted value
        setPrediction(predictionValue);
      })
      .catch(error => {
        console.error('Error predicting the image:', error);
      });
  };

  return (
    <div className="upload-container">
      <label htmlFor="file-input" className="file-input-label">
        Upload Image
      </label>
      <input
        type="file"
        id="file-input"
        onChange={handleFileChange}
        className="upload-button"
      />
      {previewUrl && (
        <div className="image-preview">
          <img src={previewUrl} alt="Preview" />
        </div>
      )}
      <button onClick={handlePredict} className="predict-button">
        Predict
      </button>
      {prediction !== '' && <p className='predection'>Prediction: {prediction}</p>}
    </div>
  );
};

export default UploadImage;

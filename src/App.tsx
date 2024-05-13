import { useState } from "react";
import "./App.css";

function App() {
  return (
    <>
      <p>Awaiting TF.js</p>
      <video autoPlay />
      <button>Enable Webcam</button>
      <button data-hot="0">Gather Class 1</button>
      <button data-hot="1">Gather Class 2</button>
      <button>Train & Predict</button>
      <button>Reset</button>
    </>
  );
}

export default App;

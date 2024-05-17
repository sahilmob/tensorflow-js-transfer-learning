import * as tf from "@tensorflow/tfjs";
import { useState, useRef } from "react";

import "./App.css";

const MOBILE_NET_INPUT_WITHD = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

function App() {
  const [tfLoaded, setTfLoaded] = useState(false);
  const [classNames, setClassNames] = useState(["0", "1"]);
  const [stopDataGather, setStopDataGather] = useState(-1);
  const videoRef = useRef<HTMLVideoElement>(null);

  const enableCamHandler = () => {
    navigator.mediaDevices
      .getUserMedia({
        video: true,
      })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        console.log(err);
      });
  };

  const trainAndPredictHandler = () => {};

  const resetHandler = () => {};

  const gatherDataHandler = (e: any) => {};

  return (
    <>
      <p>Awaiting TF.js</p>
      <video autoPlay />
      <button onClick={enableCamHandler}>Enable Webcam</button>
      <button
        data-hot="0"
        onMouseUp={gatherDataHandler}
        onMouseDown={gatherDataHandler}
      >
        Gather Class 1
      </button>
      <button
        data-hot="1"
        onMouseUp={gatherDataHandler}
        onMouseDown={gatherDataHandler}
      >
        Gather Class 2
      </button>
      <button onClick={trainAndPredictHandler}>Train & Predict</button>
      <button onClick={resetHandler}>Reset</button>
    </>
  );
}

export default App;

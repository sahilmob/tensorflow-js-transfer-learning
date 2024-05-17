import * as tf from "@tensorflow/tfjs";
import { useState, useRef, useEffect } from "react";

import "./App.css";

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

function App() {
  const [tfLoaded, setTfLoaded] = useState(false);
  const [classNames, setClassNames] = useState(["0", "1"]);
  const [stopDataGather, setStopDataGather] = useState(-1);
  const [gatherDataState, setGatherDataState] = useState("STOP_DATA_GATHER");
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [tainingDataInputs, setTrainingDataInputs] = useState([]);
  const [tainingDataOutputs, setTrainingDataOutputs] = useState([]);
  const [examplesCount, setExamplesCount] = useState([]);
  const [predict, setPredict] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const mobilenet = useRef<tf.GraphModel | null>(null);
  const modelHead = useRef<tf.Sequential | null>(null);

  useEffect(() => {
    async function loadModel() {
      try {
        const model = await tf.loadGraphModel(
          "https://www.kaggle.com/models/google/mobilenet-v3/TfJs/small-100-224-feature-vector/1",
          { fromTFHub: true }
        );
        mobilenet.current = model;
        setTfLoaded(true);
        tf.tidy(() => {
          const answer = model.predict(
            tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
          );
          console.log(answer);
        });
      } catch (e) {
        setTfLoaded(false);
        console.log(e);
      }
    }

    function createModelHead() {
      modelHead.current = tf.sequential();
      modelHead.current.add(
        tf.layers.dense({
          units: 128,
          activation: "relu",
          inputShape: [1024],
        })
      );
      modelHead.current.add(
        tf.layers.dense({
          units: classNames.length,
          activation: "softmax",
        })
      );

      modelHead.current.summary();

      modelHead.current.compile({
        optimizer: tf.train.adam(),
        loss:
          classNames.length === 2
            ? "binaryCrossentropy"
            : "categoricalCrossentropy",
        metrics: ["accuracy"],
      });
    }

    async function run() {
      await loadModel();
      createModelHead();
    }

    run();
  }, []);

  const enableCamHandler = () => {
    navigator.mediaDevices
      .getUserMedia({
        video: true,
      })
      .then((stream) => {
        if (videoRef.current) {
          setVideoPlaying(true);
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        setVideoPlaying(false);
        console.log(err);
      });
  };

  const trainAndPredictHandler = () => {};

  const resetHandler = () => {};

  const gatherDataHandler = (e: any) => {};

  return (
    <>
      {!tfLoaded && <p>Awaiting TF.js</p>}
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

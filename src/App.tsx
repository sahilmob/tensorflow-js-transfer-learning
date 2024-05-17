import * as tf from "@tensorflow/tfjs";
import { useState, useRef, useEffect, useCallback } from "react";

import "./App.css";

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

function App() {
  const [classNames] = useState(["0", "1"]);
  const [tfLoaded, setTfLoaded] = useState(false);
  const [gatherDataState, setGatherDataState] = useState<string | number>(
    "STOP_DATA_GATHER"
  );
  const [result, setResult] = useState<string>("");
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [trainingDataInputs, setTrainingDataInputs] = useState([]);
  const [trainingDataOutputs, setTrainingDataOutputs] = useState([]);
  const [examplesCount, setExamplesCount] = useState([]);
  const [predict, setPredict] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const mobilenet = useRef<tf.GraphModel | null>(null);
  const modelHead = useRef<tf.Sequential | null>(null);
  const dataGatherRequestAnimationFrameRef = useRef<number | null>(null);
  const predictRequestAnimationFrameRef = useRef<number | null>(null);

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
  }, [classNames.length]);

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

  const trainAndPredictHandler = async () => {
    setPredict(false);
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
    const outputAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
    const oneHotOutput = tf.oneHot(outputAsTensor, classNames.length);
    const inputsAsTensor = tf.stack(trainingDataInputs);

    await modelHead.current!.fit(inputsAsTensor, oneHotOutput, {
      shuffle: true,
      epochs: 10,
      batchSize: 5,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log(`Epoch: ${epoch}, Loss: ${logs?.loss}`);
        },
      },
    });

    outputAsTensor.dispose();
    oneHotOutput.dispose();
    inputsAsTensor.dispose();
    setPredict(true);
    predictRequestAnimationFrameRef.current =
      window.requestAnimationFrame(predictLoop);
  };

  const predictLoop = useCallback(() => {
    if (predict) {
      tf.tidy(() => {
        const videoFrameAsTensor = tf.browser
          .fromPixels(videoRef.current!)
          .div(255);
        const resizedTensorFrame = tf.image.resizeBilinear(
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          videoFrameAsTensor,
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
        );

        const imageFeatures = mobilenet.current!.predict(
          resizedTensorFrame.expandDims()
        );
        const prediction = modelHead
          .current!.predict(
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-ignore
            imageFeatures
          )
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          .squeeze() as tf.Tensor;
        const heightsIndex = prediction.argMax().arraySync();
        const predictionArray = prediction.arraySync();

        setResult(
          "Prediction: " +
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-ignore
            classNames[heightsIndex] +
            " with " +
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-ignore
            Math.floor(predictionArray[heightsIndex] * 100) +
            "% confidence"
        );

        predictRequestAnimationFrameRef.current =
          window.requestAnimationFrame(predictLoop);
      });
    }
  }, [predict, classNames]);

  useEffect(() => {
    if (predict) {
      predictRequestAnimationFrameRef.current =
        window.requestAnimationFrame(predictLoop);
    } else {
      window.cancelAnimationFrame(predictRequestAnimationFrameRef.current!);
    }

    return () => {
      window.cancelAnimationFrame(predictRequestAnimationFrameRef.current!);
    };
  }, [predict, predictLoop]);

  const resetHandler = () => {
    setTrainingDataInputs([]);
    setTrainingDataOutputs([]);
    setExamplesCount([]);
    setPredict(false);
    setResult("");

    window.cancelAnimationFrame(dataGatherRequestAnimationFrameRef.current!);
    window.cancelAnimationFrame(predictRequestAnimationFrameRef.current!);
  };

  const dataGatherLoop = useCallback(() => {
    if (videoPlaying && gatherDataState !== "STOP_DATA_GATHER") {
      const imageFeature = tf.tidy(() => {
        const videoFrameAsTensor = tf.browser.fromPixels(videoRef.current!);
        const resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor,
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
        );
        const normalizedTensorFrame = resizedTensorFrame.div(255);
        return (
          mobilenet
            .current!.predict(normalizedTensorFrame.expandDims())
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-expect-error
            .squeeze()
        );
      });

      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      setTrainingDataInputs((prev) => [...prev, imageFeature]);
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      setTrainingDataOutputs((prev) => [...prev, gatherDataState]);

      if (examplesCount[gatherDataState as number] === undefined) {
        setExamplesCount((prev) => {
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          prev[gatherDataState] = 0;
          return prev;
        });
      }

      setExamplesCount((prev) => {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        prev[gatherDataState] = prev[gatherDataState] + 1;
        return prev;
      });
    }

    dataGatherRequestAnimationFrameRef.current =
      window.requestAnimationFrame(dataGatherLoop);
  }, [gatherDataState, videoPlaying, examplesCount]);

  useEffect(() => {
    if (typeof gatherDataState === "number") {
      dataGatherRequestAnimationFrameRef.current =
        window.requestAnimationFrame(dataGatherLoop);
    } else {
      window.cancelAnimationFrame(dataGatherRequestAnimationFrameRef.current!);
    }

    return () => {
      window.cancelAnimationFrame(dataGatherRequestAnimationFrameRef.current!);
    };
  }, [dataGatherLoop, gatherDataState]);

  const gatherDataHandler = (cls: number) => {
    setGatherDataState((prev) =>
      prev === "STOP_DATA_GATHER" ? cls : "STOP_DATA_GATHER"
    );
  };

  return (
    <>
      {!tfLoaded && <p>Awaiting TF.js</p>}
      <p>{result}</p>
      <video autoPlay ref={videoRef} />
      {!videoPlaying && (
        <button disabled={!tfLoaded} onClick={enableCamHandler}>
          Enable Webcam
        </button>
      )}
      <button
        disabled={!videoPlaying}
        onMouseUp={() => gatherDataHandler(0)}
        onMouseDown={() => gatherDataHandler(0)}
      >
        Gather Class 1 ({examplesCount?.[0] || 0})
      </button>
      <button
        disabled={!videoPlaying}
        onMouseUp={() => gatherDataHandler(1)}
        onMouseDown={() => gatherDataHandler(1)}
      >
        Gather Class 2 ({examplesCount?.[1] || 0})
      </button>
      <button onClick={trainAndPredictHandler}>Train & Predict</button>
      <button onClick={resetHandler}>Reset</button>
    </>
  );
}

export default App;

"use client";

import { useEffect, useRef, useState } from "react";

type CartPoleState = {
  x: number;
  xDot: number;
  theta: number;
  thetaDot: number;
};

type EpisodeMetrics = {
  steps: number;
  totalReward: number;
  episodes: number;
  best: number;
  last: number;
};

const gravity = 9.8;
const massCart = 1;
const massPole = 0.1;
const totalMass = massCart + massPole;
const length = 0.5;
const poleMassLength = massPole * length;
const forceMag = 10;
const tau = 0.02; // Seconds between state updates
const trackLimit = 2.4;
const angleLimit = (12 * Math.PI) / 180;

const canvasWidth = 640;
const canvasHeight = 360;

const networkConfig = {
  inputSize: 4,
  hiddenSize: 8,
  outputSize: 2,
} as const;

const learningRate = 0.02;
const discountFactor = 0.99;

type NetworkParams = {
  W1: number[];
  b1: number[];
  W2: number[];
  b2: number[];
};

type NetworkActivations = {
  input: number[];
  hidden: number[];
  output: number[];
};

type PolicyStep = {
  state: number[];
  hidden: number[];
  probs: number[];
  actionIndex: number;
  reward: number;
};

function initializeNetwork(): NetworkParams {
  const { inputSize, hiddenSize, outputSize } = networkConfig;

  const randomWeight = () => (Math.random() * 2 - 1) * 0.3;

  return {
    W1: Array.from({ length: hiddenSize * inputSize }, randomWeight),
    b1: Array(hiddenSize).fill(0),
    W2: Array.from({ length: outputSize * hiddenSize }, randomWeight),
    b2: Array(outputSize).fill(0),
  };
}

function stateToVector(state: CartPoleState): number[] {
  return [state.x, state.xDot, state.theta, state.thetaDot];
}

function forwardNetwork(
  params: NetworkParams,
  inputVector: number[],
): { hidden: number[]; probs: number[] } {
  const { inputSize, hiddenSize, outputSize } = networkConfig;

  const hidden: number[] = new Array(hiddenSize);
  for (let i = 0; i < hiddenSize; i += 1) {
    let sum = params.b1[i];
    for (let j = 0; j < inputSize; j += 1) {
      sum += params.W1[i * inputSize + j] * inputVector[j];
    }
    hidden[i] = Math.tanh(sum);
  }

  const logits: number[] = new Array(outputSize);
  for (let o = 0; o < outputSize; o += 1) {
    let sum = params.b2[o];
    for (let h = 0; h < hiddenSize; h += 1) {
      sum += params.W2[o * hiddenSize + h] * hidden[h];
    }
    logits[o] = sum;
  }

  const maxLogit = Math.max(...logits);
  let denom = 0;
  const exps = logits.map((logit) => {
    const value = Math.exp(logit - maxLogit);
    denom += value;
    return value;
  });

  const probs = exps.map((value) => value / denom);

  return { hidden, probs };
}

function sampleActionIndex(probabilities: number[]): number {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i += 1) {
    cumulative += probabilities[i];
    if (r < cumulative) {
      return i;
    }
  }
  return probabilities.length - 1;
}

function trainOnEpisode(
  params: NetworkParams,
  trajectory: PolicyStep[],
): NetworkParams {
  if (trajectory.length === 0) {
    return params;
  }

  const { inputSize, hiddenSize, outputSize } = networkConfig;
  const gradW1 = new Array(hiddenSize * inputSize).fill(0);
  const gradb1 = new Array(hiddenSize).fill(0);
  const gradW2 = new Array(outputSize * hiddenSize).fill(0);
  const gradb2 = new Array(outputSize).fill(0);

  const returns = new Array(trajectory.length).fill(0);
  let discounted = 0;
  for (let t = trajectory.length - 1; t >= 0; t -= 1) {
    discounted = trajectory[t].reward + discountFactor * discounted;
    returns[t] = discounted;
  }

  for (let t = 0; t < trajectory.length; t += 1) {
    const step = trajectory[t];
    const advantage = returns[t];

    const delta2 = new Array(outputSize);
    for (let o = 0; o < outputSize; o += 1) {
      const indicator = step.actionIndex === o ? 1 : 0;
      delta2[o] = (indicator - step.probs[o]) * advantage;
      gradb2[o] += delta2[o];
      for (let h = 0; h < hiddenSize; h += 1) {
        gradW2[o * hiddenSize + h] += delta2[o] * step.hidden[h];
      }
    }

    const delta1 = new Array(hiddenSize).fill(0);
    for (let h = 0; h < hiddenSize; h += 1) {
      let sum = 0;
      for (let o = 0; o < outputSize; o += 1) {
        sum += params.W2[o * hiddenSize + h] * delta2[o];
      }
      delta1[h] = (1 - step.hidden[h] * step.hidden[h]) * sum;
      gradb1[h] += delta1[h];
      for (let i = 0; i < inputSize; i += 1) {
        gradW1[h * inputSize + i] += delta1[h] * step.state[i];
      }
    }
  }

  const scale = learningRate / Math.max(1, trajectory.length);

  return {
    W1: params.W1.map((weight, index) => weight + scale * gradW1[index]),
    b1: params.b1.map((bias, index) => bias + scale * gradb1[index]),
    W2: params.W2.map((weight, index) => weight + scale * gradW2[index]),
    b2: params.b2.map((bias, index) => bias + scale * gradb2[index]),
  };
}

function drawRoundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const effectiveRadius = Math.max(
    0,
    Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2),
  );
  context.beginPath();
  context.moveTo(x + effectiveRadius, y);
  context.lineTo(x + width - effectiveRadius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + effectiveRadius);
  context.lineTo(x + width, y + height - effectiveRadius);
  context.quadraticCurveTo(
    x + width,
    y + height,
    x + width - effectiveRadius,
    y + height,
  );
  context.lineTo(x + effectiveRadius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - effectiveRadius);
  context.lineTo(x, y + effectiveRadius);
  context.quadraticCurveTo(x, y, x + effectiveRadius, y);
  context.closePath();
}

function createInitialState(): CartPoleState {
  // Use deterministic values for SSR, random for client
  if (typeof window === 'undefined') {
    return {
      x: 0,
      xDot: 0,
      theta: 0,
      thetaDot: 0,
    };
  }
  return {
    x: (Math.random() - 0.5) * 0.4,
    xDot: 0,
    theta: (Math.random() - 0.5) * 0.2,
    thetaDot: 0,
  };
}

function stepCartPole(
  state: CartPoleState,
  action: -1 | 1,
): { nextState: CartPoleState; done: boolean } {
  const { x, xDot, theta, thetaDot } = state;
  const force = action * forceMag;
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);

  const temp =
    (force + poleMassLength * thetaDot * thetaDot * sinTheta) / totalMass;
  const thetaAcc =
    (gravity * sinTheta - cosTheta * temp) /
    (length * (4 / 3 - (massPole * cosTheta * cosTheta) / totalMass));
  const xAcc =
    temp - (poleMassLength * thetaAcc * cosTheta) / totalMass;

  const nextState: CartPoleState = {
    x: x + tau * xDot,
    xDot: xDot + tau * xAcc,
    theta: theta + tau * thetaDot,
    thetaDot: thetaDot + tau * thetaAcc,
  };

  const done =
    Math.abs(nextState.x) > trackLimit || Math.abs(nextState.theta) > angleLimit;

  return { nextState, done };
}

function CartPoleCanvas({ state }: { state: CartPoleState }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const devicePixelRatio = window.devicePixelRatio || 1;
    if (canvas.width !== canvasWidth * devicePixelRatio) {
      canvas.width = canvasWidth * devicePixelRatio;
      canvas.height = canvasHeight * devicePixelRatio;
      canvas.style.width = `${canvasWidth}px`;
      canvas.style.height = `${canvasHeight}px`;
      context.scale(devicePixelRatio, devicePixelRatio);
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    context.clearRect(0, 0, canvasWidth, canvasHeight);

    // Draw track
    context.fillStyle = "rgba(0, 0, 0, 0.1)";
    const trackHeight = 12;
    const trackY = canvasHeight / 2 + 70;
    context.fillRect(0, trackY, canvasWidth, trackHeight);

    // Draw safe zone markers
    context.fillStyle = "rgba(59, 130, 246, 0.25)";
    const safeZoneWidth = (canvasWidth * (trackLimit * 2)) / (trackLimit * 4.5);
    context.fillRect(
      (canvasWidth - safeZoneWidth) / 2,
      trackY,
      safeZoneWidth,
      trackHeight,
    );

    // Convert physics coordinates to canvas
    const pixelsPerMeter = canvasWidth / (trackLimit * 4);
    const cartCenterX = canvasWidth / 2 + state.x * pixelsPerMeter;
    const cartY = trackY;

    // Draw cart shadow
    context.fillStyle = "rgba(15, 23, 42, 0.15)";
    context.beginPath();
    context.ellipse(cartCenterX, cartY + 20, 55, 12, 0, 0, Math.PI * 2);
    context.fill();

    // Draw cart
    const cartWidth = 120;
    const cartHeight = 40;
    context.fillStyle = "#0f172a";
    drawRoundedRect(
      context,
      cartCenterX - cartWidth / 2,
      cartY - cartHeight,
      cartWidth,
      cartHeight,
      12,
    );
    context.fill();

    // Draw pole
    const poleLength = 140;
    const poleWidth = 12;
    const poleAngle = state.theta;
    context.save();
    context.translate(cartCenterX, cartY - cartHeight);
    context.rotate(poleAngle);
    const gradient = context.createLinearGradient(0, -poleLength, 0, 0);
    gradient.addColorStop(0, "#36cfc9");
    gradient.addColorStop(1, "#2563eb");
    context.fillStyle = gradient;
    drawRoundedRect(
      context,
      -poleWidth / 2,
      -poleLength,
      poleWidth,
      poleLength,
      6,
    );
    context.fill();

    // Draw pole tip weight
    context.beginPath();
    context.fillStyle = "#1d4ed8";
    context.arc(0, -poleLength, 14, 0, Math.PI * 2);
    context.fill();
    context.restore();

    // Draw axle
    context.fillStyle = "#f8fafc";
    context.beginPath();
    context.arc(cartCenterX, cartY - cartHeight, 8, 0, Math.PI * 2);
    context.fill();

    // Draw wheels
    context.fillStyle = "#1e293b";
    const wheelRadius = 14;
    const wheelOffset = 36;
    context.beginPath();
    context.arc(cartCenterX - wheelOffset, cartY - 2, wheelRadius, 0, Math.PI * 2);
    context.arc(cartCenterX + wheelOffset, cartY - 2, wheelRadius, 0, Math.PI * 2);
    context.fill();

    // Draw angle guide
    context.strokeStyle = "rgba(59, 130, 246, 0.35)";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(cartCenterX, cartY - cartHeight);
    context.lineTo(cartCenterX, cartY - cartHeight - poleLength - 10);
    context.stroke();
  }, [state]);

  return (
    <canvas
      ref={canvasRef}
      width={canvasWidth}
      height={canvasHeight}
      className="rounded-3xl bg-white/70 shadow-xl backdrop-blur-md"
    />
  );
}

export default function Home() {
  const [isRunning, setIsRunning] = useState(false);
  const [controlMode, setControlMode] = useState<
    "neural" | "heuristic" | "random"
  >("neural");
  const [state, setState] = useState<CartPoleState>(() => createInitialState());
  const stateRef = useRef(state);
  const [metrics, setMetrics] = useState<EpisodeMetrics>(
    () => ({ steps: 0, totalReward: 0, episodes: 0, best: 0, last: 0 }),
  );
  const animationRef = useRef<number | null>(null);
  const [network, setNetwork] = useState<NetworkParams | null>(null);
  const networkRef = useRef<NetworkParams | null>(null);
  const [activations, setActivations] = useState<NetworkActivations>(() => ({
    input: Array(networkConfig.inputSize).fill(0),
    hidden: Array(networkConfig.hiddenSize).fill(0),
    output: Array(networkConfig.outputSize).fill(0),
  }));
  const trajectoryRef = useRef<PolicyStep[]>([]);
  const [trainingStats, setTrainingStats] = useState({
    averageReturn: 0,
    lastReturn: 0,
  });
  const [isClient, setIsClient] = useState(false);

  // Initialize network and set client flag only on client side
  useEffect(() => {
    setIsClient(true);
    const initialNetwork = initializeNetwork();
    setNetwork(initialNetwork);
    networkRef.current = initialNetwork;
    
    // Re-initialize state with random values on client side
    const clientState = createInitialState();
    setState(clientState);
    stateRef.current = clientState;
  }, []);

  useEffect(() => {
    networkRef.current = network;
  }, [network]);


  useEffect(() => {
    if (controlMode !== "neural") {
      trajectoryRef.current = [];
    }
  }, [controlMode]);

  useEffect(() => {
    if (!isRunning) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      return;
    }

    const loop = () => {
      const current = stateRef.current;
      const stateVector = stateToVector(current);
      let action: -1 | 1 = -1;

      if (controlMode === "heuristic") {
        action =
          current.theta + 0.25 * current.thetaDot + 0.05 * current.xDot > 0
            ? 1
            : -1;
      } else if (controlMode === "random") {
        action = Math.random() > 0.5 ? 1 : -1;
      } else if (networkRef.current) {
        const { probs, hidden } = forwardNetwork(
          networkRef.current,
          stateVector,
        );
        const actionIndex = sampleActionIndex(probs);
        action = actionIndex === 0 ? -1 : 1;

        trajectoryRef.current.push({
          state: [...stateVector],
          hidden,
          probs,
          actionIndex,
          reward: 1,
        });

        const newActivations: NetworkActivations = {
          input: stateVector,
          hidden,
          output: probs,
        };
        setActivations(newActivations);
      } else {
        // Fallback to heuristic if network not ready
        action =
          current.theta + 0.25 * current.thetaDot + 0.05 * current.xDot > 0
            ? 1
            : -1;
      }

      const { nextState, done } = stepCartPole(current, action);

      if (done) {
        setMetrics((prev) => {
          const finishedSteps = prev.steps + 1;
          return {
            steps: 0,
            totalReward: 0,
            episodes: prev.episodes + 1,
            best: Math.max(prev.best, finishedSteps),
            last: finishedSteps,
          } satisfies EpisodeMetrics;
        });
        if (controlMode === "neural" && networkRef.current) {
          const episodeReward = trajectoryRef.current.length;
          const updatedNetwork = trainOnEpisode(
            networkRef.current,
            trajectoryRef.current,
          );
          trajectoryRef.current = [];
          networkRef.current = updatedNetwork;
          setNetwork(updatedNetwork);
          setTrainingStats((prev) => {
            const smoothing =
              prev.averageReturn === 0
                ? episodeReward
                : prev.averageReturn * 0.9 + episodeReward * 0.1;
            return {
              averageReturn: smoothing,
              lastReturn: episodeReward,
            };
          });
        }
        const resetState = createInitialState();
        stateRef.current = resetState;
        setState(resetState);
      } else {
        stateRef.current = nextState;
        setState(nextState);
        setMetrics((prev) => ({
          steps: prev.steps + 1,
          totalReward: prev.totalReward + 1,
          episodes: prev.episodes,
          best: prev.best,
          last: prev.last,
        }));
      }

      animationRef.current = requestAnimationFrame(loop);
    };

    animationRef.current = requestAnimationFrame(loop);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [isRunning, controlMode]);

  const toggleRun = () => {
    setIsRunning((prev) => !prev);
  };

  const handleReset = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    const resetState = createInitialState();
    stateRef.current = resetState;
    setState(resetState);
    setIsRunning(false);
    trajectoryRef.current = [];
    setMetrics((prev) => ({
      steps: 0,
      totalReward: 0,
      episodes: prev.episodes,
      best: prev.best,
      last: prev.last,
    }));
  };

  const handleReinitializeNetwork = () => {
    const freshNetwork = initializeNetwork();
    trajectoryRef.current = [];
    setTrainingStats({ averageReturn: 0, lastReturn: 0 });
    setActivations({
      input: Array(networkConfig.inputSize).fill(0),
      hidden: Array(networkConfig.hiddenSize).fill(0),
      output: Array(networkConfig.outputSize).fill(0),
    });
    networkRef.current = freshNetwork;
    setNetwork(freshNetwork);
  };

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_20%_20%,#dbeafe,transparent_60%),radial-gradient(circle_at_80%_0,#c7d2fe,transparent_50%),radial-gradient(circle_at_0_80%,#bae6fd,transparent_55%)] px-6 py-16 text-slate-900">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-12">
        <header className="flex flex-col gap-6 text-center md:text-left">
          <span className="mx-auto inline-flex items-center gap-2 rounded-full bg-white/60 px-4 py-2 text-sm font-medium text-sky-700 shadow-sm backdrop-blur-md md:mx-0">
            <span className="h-2 w-2 rounded-full bg-sky-500" />
            リアルタイム強化学習プレイグラウンド
          </span>
          <h1 className="text-4xl font-semibold text-slate-900 md:text-5xl">
            CartPole ビジュアルサンドボックス
          </h1>
          <div className="space-y-4 text-slate-600 md:max-w-3xl">
            <p className="text-lg">
              <strong>CartPole問題</strong>は、強化学習の入門として最適な制御問題です。
              倒立振り子（逆さまに立てた棒）をカートの上でバランスを取り続けることが目標です。
            </p>
            <div className="grid gap-4 text-sm md:grid-cols-2">
              <div className="rounded-lg bg-blue-50/80 p-4">
                <h3 className="font-semibold text-blue-900 mb-2">🎯 学習目標</h3>
                <ul className="space-y-1 text-blue-800">
                  <li>• 強化学習の基本概念を理解</li>
                  <li>• AIの学習過程を視覚的に観察</li>
                  <li>• 制御理論と物理シミュレーション</li>
                </ul>
              </div>
              <div className="rounded-lg bg-green-50/80 p-4">
                <h3 className="font-semibold text-green-900 mb-2">🔬 観察ポイント</h3>
                <ul className="space-y-1 text-green-800">
                  <li>• ニューラルネットワークの重み変化</li>
                  <li>• 学習による性能向上</li>
                  <li>• 異なる制御方法の比較</li>
                </ul>
              </div>
            </div>
          </div>
        </header>

        <section className="grid gap-10 rounded-3xl bg-white/60 p-8 shadow-xl ring-1 ring-slate-200 backdrop-blur-md md:grid-cols-[2fr_1fr]">
          <div className="flex flex-col gap-6">
            <CartPoleCanvas state={state} />
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={toggleRun}
                className="inline-flex items-center justify-center gap-2 rounded-full bg-slate-900 px-6 py-2 text-sm font-semibold text-white shadow-lg transition hover:bg-slate-700"
              >
                {isRunning ? "一時停止" : "開始"}
              </button>
              <button
                onClick={handleReset}
                className="inline-flex items-center justify-center gap-2 rounded-full border border-slate-200 px-6 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-white"
              >
                リセット
              </button>
              <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-medium text-slate-600 shadow-sm">
                <span className="text-slate-500">制御方法</span>
                <div className="flex overflow-hidden rounded-full border border-slate-200">
                  {[
                    { key: "neural", label: "ニューラル" },
                    { key: "heuristic", label: "ヒューリスティック" },
                    { key: "random", label: "ランダム" },
                  ].map((option) => (
                    <button
                      key={option.key}
                      onClick={() => setControlMode(option.key as typeof controlMode)}
                      className={`px-4 py-1 text-sm transition ${
                        controlMode === option.key
                          ? "bg-slate-900 text-white"
                          : "bg-white text-slate-600 hover:bg-slate-100"
                      }`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              {controlMode === "neural" && (
                <button
                  onClick={handleReinitializeNetwork}
                  className="inline-flex items-center justify-center gap-2 rounded-full border border-slate-200 px-5 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-white"
                >
                  ネットワークリセット
                </button>
              )}
            </div>
          </div>

          <aside className="flex flex-col gap-6">
            <div className="rounded-2xl border border-white/50 bg-white/80 p-6 shadow-sm">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                エピソードメトリクス
              </h2>
              <div className="mt-4 space-y-3 text-sm">
                <MetricRow
                  label="現在のエピソードのステップ数"
                  value={metrics.steps.toString()}
                />
                <MetricRow
                  label="現在のエピソードの報酬"
                  value={metrics.totalReward.toString()}
                />
                <MetricRow
                  label="完了したエピソード数"
                  value={metrics.episodes.toString()}
                />
                <MetricRow
                  label="前回の生存時間"
                  value={`${metrics.last} ステップ`}
                />
                <MetricRow
                  label="最長生存時間"
                  value={`${metrics.best} ステップ`}
                />
              </div>
            </div>
            <div className="rounded-2xl border border-white/50 bg-white/90 p-6 shadow-sm">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                ニューラル方策エクスプローラー
              </h2>
              <div className="mt-4">
                {isClient && network ? (
                  <NetworkVisualizer params={network} activations={activations} />
                ) : (
                  <div className="flex h-56 w-full items-center justify-center rounded-lg bg-slate-100">
                    <span className="text-sm text-slate-500">ネットワーク読み込み中...</span>
                  </div>
                )}
              </div>
              <div className="mt-4 space-y-2 text-sm">
                <MetricRow
                  label="前回エピソードの報酬"
                  value={trainingStats.lastReturn.toFixed(0)}
                />
                <MetricRow
                  label="移動平均報酬"
                  value={trainingStats.averageReturn.toFixed(1)}
                />
                <MetricRow
                  label="学習率"
                  value={learningRate.toString()}
                />
              </div>
              <div className="mt-4 space-y-3">
                <p className="text-xs text-slate-500">
                  重みが強くなる（青）か弱くなる（ローズ）に従って、接続線の太さや色が変化します。
                  ノードの塗りつぶしは最新の活性化を示しています。
                </p>
                <div className="rounded-lg bg-slate-50/80 p-3 text-xs">
                  <h4 className="font-semibold text-slate-700 mb-2">📊 ネットワーク構造の説明</h4>
                  <div className="space-y-1 text-slate-600">
                    <p><strong>入力層（4ノード）：</strong>カートの位置、速度、ポールの角度、角速度</p>
                    <p><strong>隠れ層（8ノード）：</strong>入力を組み合わせて特徴を抽出</p>
                    <p><strong>出力層（2ノード）：</strong>左右の行動確率（ソフトマックス関数）</p>
                    <p><strong>学習：</strong>REINFORCE法で方策勾配を計算し重みを更新</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="rounded-2xl border border-white/40 bg-gradient-to-br from-sky-500/20 via-indigo-500/20 to-blue-600/40 p-6 text-sm text-slate-600 shadow-sm">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                仕組み
              </h2>
              <div className="mt-3 space-y-4">
                <p>
                  CartPole環境はカートに左右の力を加えてポールのバランスを取ります。
                  ヒューリスティックコントローラーはポールの角度と速度を使って
                  どちらの方向に押すかを決定します。
                </p>
                
                <div className="space-y-3 text-xs">
                  <div className="rounded-md bg-white/50 p-3">
                    <h4 className="font-semibold text-slate-700 mb-2">🧮 物理の数式</h4>
                    <div className="space-y-1 text-slate-600 font-mono">
                      <p>カートの加速度: a_cart = (F - m_pole × L × θ̈ × cos(θ)) / m_total</p>
                      <p>ポールの角加速度: θ̈ = (g × sin(θ) - a_cart × cos(θ)) / L</p>
                      <p>重力: g = 9.8 m/s², カート質量: 1kg, ポール質量: 0.1kg</p>
                    </div>
                  </div>
                  
                  <div className="rounded-md bg-white/50 p-3">
                    <h4 className="font-semibold text-slate-700 mb-2">🎯 制御方法の比較</h4>
                    <div className="space-y-1 text-slate-600">
                      <p><strong>ヒューリスティック：</strong>if (θ + 0.25×θ̇ + 0.05×ẋ &gt; 0) then 右 else 左</p>
                      <p><strong>ニューラル：</strong>NN(x, ẋ, θ, θ̇) → [P(左), P(右)] → 確率的選択</p>
                      <p><strong>ランダム：</strong>50%の確率で左右をランダム選択</p>
                    </div>
                  </div>
                  
                  <div className="rounded-md bg-white/50 p-3">
                    <h4 className="font-semibold text-slate-700 mb-2">📈 学習アルゴリズム（REINFORCE）</h4>
                    <div className="space-y-1 text-slate-600">
                      <p>1. エピソード実行：状態観察 → 行動選択 → 報酬獲得</p>
                      <p>2. 収益計算：R_t = Σ(γ^k × r_&#123;t+k&#125;) (γ=0.99)</p>
                      <p>3. 勾配更新：∇θ = Σ(R_t × ∇log π(a_t|s_t))</p>
                      <p>4. 重み更新：θ ← θ + α × ∇θ (α=0.02)</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </aside>
        </section>
      </div>
    </div>
  );
}

function NetworkVisualizer({
  params,
  activations,
}: {
  params: NetworkParams;
  activations: NetworkActivations;
}) {
  const { inputSize, hiddenSize, outputSize } = networkConfig;
  const width = 360;
  const height = 220;
  const marginX = 36;
  const columnSpacing = (width - marginX * 2) / 2;
  const nodeRadius = 16;

  const inputPositions = Array.from({ length: inputSize }, (_, index) => ({
    x: marginX,
    y: ((index + 1) / (inputSize + 1)) * height,
  }));

  const hiddenPositions = Array.from({ length: hiddenSize }, (_, index) => ({
    x: marginX + columnSpacing,
    y: ((index + 1) / (hiddenSize + 1)) * height,
  }));

  const outputPositions = Array.from({ length: outputSize }, (_, index) => ({
    x: marginX + columnSpacing * 2,
    y: ((index + 1) / (outputSize + 1)) * height,
  }));

  const inputLabels = ["x", "xDot", "theta", "thetaDot"];
  const outputLabels = ["左", "右"];

  const weightColor = (weight: number) => {
    const normalized = Math.tanh(weight);
    const intensity = Math.abs(normalized);
    const alpha = 0.2 + intensity * 0.6;
    return normalized >= 0
      ? `rgba(59, 130, 246, ${alpha})`
      : `rgba(244, 63, 94, ${alpha})`;
  };

  const weightThickness = (weight: number) => {
    const normalized = Math.abs(Math.tanh(weight));
    return 1.5 + normalized * 4;
  };

  const activationColor = (
    value: number,
    kind: "input" | "hidden" | "output",
  ) => {
    if (kind === "output") {
      const clamped = Math.min(1, Math.max(0, value));
      const alpha = 0.25 + clamped * 0.6;
      return `rgba(16, 185, 129, ${alpha})`;
    }
    const clamped = Math.max(-1, Math.min(1, value));
    const intensity = Math.abs(clamped);
    const alpha = 0.25 + intensity * 0.55;
    return clamped >= 0
      ? `rgba(37, 99, 235, ${alpha})`
      : `rgba(248, 113, 113, ${alpha})`;
  };

  const inputValues = activations.input.map((value) => Math.tanh(value));
  const hiddenValues = activations.hidden;
  const outputValues = activations.output;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-56 w-full">
      {[...Array(hiddenSize)].flatMap((_, h) =>
        inputPositions.map((inputPos, i) => {
          const weight = params.W1[h * inputSize + i];
          const hiddenPos = hiddenPositions[h];
          return (
            <line
              key={`in-${i}-hid-${h}`}
              x1={inputPos.x + nodeRadius}
              y1={inputPos.y}
              x2={hiddenPos.x - nodeRadius}
              y2={hiddenPos.y}
              stroke={weightColor(weight)}
              strokeWidth={weightThickness(weight)}
              strokeLinecap="round"
            />
          );
        }),
      )}
      {[...Array(outputSize)].flatMap((_, o) =>
        hiddenPositions.map((hiddenPos, h) => {
          const weight = params.W2[o * hiddenSize + h];
          const outputPos = outputPositions[o];
          return (
            <line
              key={`hid-${h}-out-${o}`}
              x1={hiddenPos.x + nodeRadius}
              y1={hiddenPos.y}
              x2={outputPos.x - nodeRadius}
              y2={outputPos.y}
              stroke={weightColor(weight)}
              strokeWidth={weightThickness(weight)}
              strokeLinecap="round"
            />
          );
        }),
      )}
      {inputPositions.map((pos, index) => (
        <g key={`input-${index}`}>
          <circle
            cx={pos.x}
            cy={pos.y}
            r={nodeRadius}
            fill={activationColor(inputValues[index] ?? 0, "input")}
            stroke="rgba(15, 23, 42, 0.12)"
            strokeWidth={1}
          />
          <text
            x={pos.x}
            y={pos.y - nodeRadius - 8}
            textAnchor="middle"
            fontSize={11}
            fill="#64748b"
          >
            {inputLabels[index] ?? `s${index}`}
          </text>
          <text
            x={pos.x}
            y={pos.y + 4}
            textAnchor="middle"
            fontSize={11}
            fill="#0f172a"
          >
            {(activations.input[index] ?? 0).toFixed(2)}
          </text>
        </g>
      ))}
      {hiddenPositions.map((pos, index) => (
        <g key={`hidden-${index}`}>
          <circle
            cx={pos.x}
            cy={pos.y}
            r={nodeRadius}
            fill={activationColor(hiddenValues[index] ?? 0, "hidden")}
            stroke="rgba(15, 23, 42, 0.12)"
            strokeWidth={1}
          />
          <text
            x={pos.x}
            y={pos.y - nodeRadius - 6}
            textAnchor="middle"
            fontSize={11}
            fill="#64748b"
          >
            h{index + 1}
          </text>
          <text
            x={pos.x}
            y={pos.y + 4}
            textAnchor="middle"
            fontSize={11}
            fill="#0f172a"
          >
            {(hiddenValues[index] ?? 0).toFixed(2)}
          </text>
        </g>
      ))}
      {outputPositions.map((pos, index) => (
        <g key={`output-${index}`}>
          <circle
            cx={pos.x}
            cy={pos.y}
            r={nodeRadius}
            fill={activationColor(outputValues[index] ?? 0, "output")}
            stroke="rgba(15, 23, 42, 0.12)"
            strokeWidth={1}
          />
          <text
            x={pos.x}
            y={pos.y - nodeRadius - 6}
            textAnchor="middle"
            fontSize={11}
            fill="#64748b"
          >
            {outputLabels[index] ?? `a${index}`}
          </text>
          <text
            x={pos.x}
            y={pos.y + 4}
            textAnchor="middle"
            fontSize={11}
            fill="#0f172a"
          >
            {(outputValues[index] ?? 0).toFixed(2)}
          </text>
        </g>
      ))}
    </svg>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-500">{label}</span>
      <span className="font-semibold text-slate-900">{value}</span>
    </div>
  );
}

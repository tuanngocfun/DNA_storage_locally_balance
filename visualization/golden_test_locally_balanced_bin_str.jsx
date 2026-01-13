import React, { useState } from 'react';
import { Play, RefreshCw, CheckCircle, XCircle } from 'lucide-react';

const LocallyBalancedVisualizer = () => {
  const [binaryString, setBinaryString] = useState('01101001');
  const [l, setL] = useState(4);
  const [delta, setDelta] = useState(1);
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentWindow, setCurrentWindow] = useState(-1);
  const [results, setResults] = useState([]);
  const [finalResult, setFinalResult] = useState(null);

  const checkLocallyBalanced = (s, windowLen, deltaVal) => {
    const bits = s.split('').map(c => parseInt(c));
    const n = bits.length;
    const minW = windowLen / 2 - deltaVal;
    const maxW = windowLen / 2 + deltaVal;
    const windowResults = [];

    for (let i = 0; i <= n - windowLen; i++) {
      const window = bits.slice(i, i + windowLen);
      const weight = window.reduce((sum, bit) => sum + bit, 0);
      const isValid = weight >= minW && weight <= maxW;
      windowResults.push({
        start: i,
        end: i + windowLen - 1,
        window,
        weight,
        isValid,
        minW,
        maxW
      });
    }

    return windowResults;
  };

  const animate = async () => {
    setIsAnimating(true);
    setCurrentWindow(-1);
    setFinalResult(null);
    
    const windowResults = checkLocallyBalanced(binaryString, l, delta);
    setResults(windowResults);

    for (let i = 0; i < windowResults.length; i++) {
      setCurrentWindow(i);
      await new Promise(resolve => setTimeout(resolve, 800));
    }

    const allValid = windowResults.every(r => r.isValid);
    setFinalResult(allValid);
    setCurrentWindow(-1);
    setIsAnimating(false);
  };

  const generateRandom = () => {
    const length = Math.floor(Math.random() * 8) + 8; // 8-15 bits
    const random = Array.from({length}, () => Math.random() > 0.5 ? '1' : '0').join('');
    setBinaryString(random);
    setResults([]);
    setFinalResult(null);
    setCurrentWindow(-1);
  };

  const currentResult = currentWindow >= 0 ? results[currentWindow] : null;

  return (
    <div className="w-full max-w-6xl mx-auto p-8 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl">
      <h1 className="text-4xl font-bold text-center mb-2 text-indigo-900">
        Locally Balanced Binary Strings
      </h1>
      <p className="text-center text-gray-600 mb-8">
        A string is (l, δ)-locally balanced if every window of length l has between l/2-δ and l/2+δ ones
      </p>

      {/* Controls */}
      <div className="bg-white rounded-lg p-6 shadow-lg mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Binary String
            </label>
            <input
              type="text"
              value={binaryString}
              onChange={(e) => {
                if (/^[01]*$/.test(e.target.value)) {
                  setBinaryString(e.target.value);
                  setResults([]);
                  setFinalResult(null);
                }
              }}
              className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg font-mono text-lg"
              placeholder="Enter 0s and 1s"
            />
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Window Length (l) = {l}
            </label>
            <input
              type="range"
              min="2"
              max={Math.min(binaryString.length, 12)}
              value={l}
              onChange={(e) => {
                setL(parseInt(e.target.value));
                setResults([]);
                setFinalResult(null);
              }}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Delta (δ) = {delta}
            </label>
            <input
              type="range"
              min="0"
              max="5"
              value={delta}
              onChange={(e) => {
                setDelta(parseInt(e.target.value));
                setResults([]);
                setFinalResult(null);
              }}
              className="w-full"
            />
          </div>
        </div>

        <div className="flex gap-3 justify-center">
          <button
            onClick={animate}
            disabled={isAnimating || binaryString.length < l}
            className="flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            <Play size={20} />
            Check Balance
          </button>
          <button
            onClick={generateRandom}
            disabled={isAnimating}
            className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 disabled:bg-gray-400 transition-colors"
          >
            <RefreshCw size={20} />
            Random String
          </button>
        </div>
      </div>

      {/* Constraint Display */}
      <div className="bg-white rounded-lg p-6 shadow-lg mb-6">
        <h3 className="text-lg font-bold text-gray-800 mb-3">Constraint Rules:</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="text-sm text-gray-600">Window Length</div>
            <div className="text-3xl font-bold text-blue-600">{l}</div>
          </div>
          <div className="bg-green-50 rounded-lg p-4">
            <div className="text-sm text-gray-600">Min Ones Allowed</div>
            <div className="text-3xl font-bold text-green-600">{l/2 - delta}</div>
          </div>
          <div className="bg-orange-50 rounded-lg p-4">
            <div className="text-sm text-gray-600">Max Ones Allowed</div>
            <div className="text-3xl font-bold text-orange-600">{l/2 + delta}</div>
          </div>
        </div>
      </div>

      {/* Binary String Visualization */}
      {binaryString && (
        <div className="bg-white rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-lg font-bold text-gray-800 mb-4">Binary String:</h3>
          <div className="flex flex-wrap gap-2 justify-center">
            {binaryString.split('').map((bit, idx) => {
              const isInWindow = currentResult && 
                idx >= currentResult.start && 
                idx <= currentResult.end;
              
              return (
                <div
                  key={idx}
                  className={`w-12 h-12 flex items-center justify-center text-2xl font-bold rounded-lg transition-all duration-300 ${
                    isInWindow
                      ? currentResult.isValid
                        ? 'bg-green-500 text-white scale-110 shadow-lg'
                        : 'bg-red-500 text-white scale-110 shadow-lg'
                      : bit === '1'
                      ? 'bg-indigo-100 text-indigo-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {bit}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Current Window Info */}
      {currentResult && (
        <div className={`rounded-lg p-6 shadow-lg mb-6 transition-all ${
          currentResult.isValid ? 'bg-green-50 border-2 border-green-500' : 'bg-red-50 border-2 border-red-500'
        }`}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold">
              Window {currentWindow + 1} of {results.length}
            </h3>
            {currentResult.isValid ? (
              <CheckCircle className="text-green-600" size={32} />
            ) : (
              <XCircle className="text-red-600" size={32} />
            )}
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-sm text-gray-600">Position</div>
              <div className="text-2xl font-bold">{currentResult.start}-{currentResult.end}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Window</div>
              <div className="text-2xl font-bold font-mono">{currentResult.window.join('')}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Ones Count</div>
              <div className={`text-2xl font-bold ${currentResult.isValid ? 'text-green-600' : 'text-red-600'}`}>
                {currentResult.weight}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Valid Range</div>
              <div className="text-2xl font-bold text-blue-600">
                {currentResult.minW}-{currentResult.maxW}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Final Result */}
      {finalResult !== null && (
        <div className={`rounded-lg p-8 shadow-lg text-center transition-all ${
          finalResult ? 'bg-green-100 border-4 border-green-500' : 'bg-red-100 border-4 border-red-500'
        }`}>
          <div className="flex items-center justify-center gap-4 mb-4">
            {finalResult ? (
              <CheckCircle className="text-green-600" size={48} />
            ) : (
              <XCircle className="text-red-600" size={48} />
            )}
            <h2 className={`text-3xl font-bold ${finalResult ? 'text-green-800' : 'text-red-800'}`}>
              {finalResult ? 'LOCALLY BALANCED ✓' : 'NOT LOCALLY BALANCED ✗'}
            </h2>
          </div>
          <p className="text-lg text-gray-700">
            {finalResult 
              ? `All ${results.length} windows satisfy the (${l}, ${delta})-balance constraint`
              : `Failed: At least one window violates the constraint`
            }
          </p>
        </div>
      )}

      {/* Test Suite Info */}
      <div className="bg-white rounded-lg p-6 shadow-lg mt-6">
        <h3 className="text-lg font-bold text-gray-800 mb-3">About the Test Suite Generator:</h3>
        <ul className="space-y-2 text-gray-700">
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 font-bold">•</span>
            <span>Generates random binary strings of length 10-20 bits</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 font-bold">•</span>
            <span>Tests two configurations: (l=4, δ=1) and (l=8, δ=1)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 font-bold">•</span>
            <span>Aims for 50% valid and 50% invalid test cases</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 font-bold">•</span>
            <span>Outputs results as JSON with detailed metadata</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default LocallyBalancedVisualizer;
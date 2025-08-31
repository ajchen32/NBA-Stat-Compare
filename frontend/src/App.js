import React, { useState } from "react";
import "./App.css";

function App() {
  const [player, setPlayer] = useState("");
  const [date1, setDate1] = useState("");
  const [date2, setDate2] = useState("");

  const handleSubmit = () => {
    console.log("Player:", player);
    console.log("Date 1:", date1);
    console.log("Date 2:", date2);
    // later send this data to backend
  }; 

  return (
    <div className="p-6 max-w-lg mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-center">
          NBA Player Score Prediction
        </h1>
      </header>

      {/* Player Name Input */}
      <div className="mb-4">
        <label className="block mb-2 font-medium">Player Name</label>
        <input
          type="text"
          value={player}
          onChange={(e) => setPlayer(e.target.value)}
          placeholder="Enter player name"
          className="w-full border rounded-lg p-2"
        />
      </div>

      {/* Side by side dates */}
      {/* Side by side dates */}
<div className="date-container">
  <div>
    <label>Date 1</label>
    <input
      type="date"
      value={date1}
      onChange={(e) => setDate1(e.target.value)}
    />
  </div>
  <div>
    <label>Date 2</label>
    <input
      type="date"
      value={date2}
      onChange={(e) => setDate2(e.target.value)}
    />
  </div>
</div>


      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        className="w-full bg-blue-600 text-white font-semibold py-2 rounded-lg hover:bg-blue-700"
      >
        Submit
      </button>
    </div>
  );
}

export default App;

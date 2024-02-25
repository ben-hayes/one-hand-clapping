import React, { useState, useEffect } from 'react';
import './App.css';
// import Select from 'react-select'
import Layout from "./components/layout.js";
import ParamChecker from "./components/paramChecker.js"

const options = [
  {value: 'Default VST Name', label:'Default VST Name'},
  { value: 'dexed', label: 'dexed' },
  { value: 'some other vst', label: 'another' },
  { value: 'one more', label: 'more' }
]

const Dropdown = () => (
  <select id="VSTs" name="vsts">
					<option value="A">AA</option>
					<option value="B">BB</option>
					<option value="C">CC</option>
					<option value="D">DD</option>
				</select>
)

function App() {
  const [currentVST, setCurrentVST] = useState(0);
  useEffect(() => {
    fetch('/vst').then(res => res.json()).then(data => {
      setCurrentVST(data);
    });
  }, []);

  return (
    <Layout name="about">
      <h2>Text to VST-i Parameters</h2>
      <p> This program estimates VST parameters based on text-prompt input </p>

      {/* VST Loader */}
      <p>Current VST: {currentVST.name}.</p>
      <p>Current Params: <ParamChecker></ParamChecker>{currentVST.params} </p>
      
      <form action="{{url_for('load')}}" method="post">
				<label for="VSTs">Choose a VST:</label>
        {/* <Select options={options} defaultValue={currentVST}/> */}
				{Dropdown}
        <testObj/>
				<button type="submit" class="btn btn-primary btn-block"> Load VST </button>
			</form>

      {/* prediction text */}
			<form action="/predict" method="post">
				<input type="text" name="prompt" placeholder="Text Prompt" required="required"></input>
				<button type="submit" class="btn btn-primary btn-block btn-large"> Get Preset </button>
			</form>
    </Layout>
  );
}

export default App;
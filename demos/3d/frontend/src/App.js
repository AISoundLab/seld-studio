import React, { Component } from 'react';
import './App.css';
import Plot from 'react-plotly.js'
import axios from 'axios';

class App extends Component {

  // Constructor
  constructor() {
    super()
    this.state = {
      predictions: [{text: 'NOTHING', x:0,  y:0, z:0}],
      groundTruth: [{text: 'NOTHING', x:0,  y:0, z:0}]
    }
    this.onAudioFileChosen = this.onAudioFileChosen.bind(this)
    this.onGroundTruthFileChosen = this.onGroundTruthFileChosen.bind(this)
    this.uploadAudioHandler = this.uploadAudioHandler.bind(this)
    this.uploadGroundTruthHandler = this.uploadGroundTruthHandler.bind(this)
  }

    // Event handler when audio file is chosen
    onAudioFileChosen(event) {
      const file = event.target.files[0]
      if (!file) {
        return
      }

      this.setState({audioFile: file})
    }

    // Event handler when file is chosen
    onGroundTruthFileChosen(event) {
      const file = event.target.files[0]
      if (!file) {
        return
      }

      this.setState({groundTruthFile: file})
    }

    // Function for sending image to the backend
    uploadAudioHandler(e) {
    const self = this;
    const formData = new FormData();
    formData.append('file', this.state.audioFile, 'audio.wav')
    axios.post('http://localhost:5000/upload_wav', formData)
    .then((response) => {
          console.log(response.data)
          self.setState({predictions:response.data})
        })
    }

    uploadGroundTruthHandler(e) {
      const self = this;
      const formData = new FormData();
      formData.append('file', this.state.groundTruthFile, 'ground_truth.csv')
      axios.post('http://localhost:5000/upload_csv', formData)
      .then((response) => {
            console.log(response.data)
            self.setState({groundTruth:response.data})
          })
      }

  render() {
    return (
      <div className="App">
        <header className="App-header">
        <div className="App-upload">
          <div>
            <label>
            Audio:    
              <input type="file" name="file" onChange={this.onAudioFileChosen} />
              <input type="submit" onClick={this.uploadAudioHandler} />
            </label>
          </div>
          <div>
            <label>Ground Truth:  
              <input type="file" name="file" onChange={this.onGroundTruthFileChosen} />
              <input type="submit" onClick={this.uploadGroundTruthHandler} />
            </label>
          </div>
          <div>
          { 
            <Plot
            data={[
              {
                text: this.state.groundTruth.map(e => e['class']),
                x: this.state.groundTruth.map(e => e['x']),
                y: this.state.groundTruth.map(e => e['y']),
                z: this.state.groundTruth.map(e => e['z']),
                type: 'scatter3d',
                mode: 'markers',
                marker: {
                  opacity: 0.8,
                  color: 'red',
                }
              },
              {
                text: this.state.predictions.map(e => e['class']),
                x: this.state.predictions.map(e => e['x']),
                y: this.state.predictions.map(e => e['y']),
                z: this.state.predictions.map(e => e['z']),
                type: 'scatter3d',
                mode: 'markers',
                marker: {
                  opacity: 0.8,
                  color: 'blue',
                }
              },
            ]
          }
            layout={ { 
              title: 'Predictions', 
            }
          }
          />
          }
          </div>
          </div>
        </header>
      </div>
    );
  }
}

export default App;

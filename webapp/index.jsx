import React, {Component} from 'react';
import ReactDOM from 'react-dom';
import './App.css';
import SocketIO from 'socket.io-client';
import InputContainer from './input';
import HeatMap from './heat-map';


class App extends Component {

  constructor(props) {
    super(props);

    this.state = {
      data: null,
      jobTypes: [],
    };

    this.socket = new SocketIO('localhost:1234');
    this.socket.on('message', (msg) => console.log(msg));
    this.socket.on('jobTypes', types => this.setState({jobTypes: types}));
    this.socket.emit('gimmeData');
  }

  sendMsg = (data) => {
    this.socket.emit('request', data);
  };

  render() {
    return (
      <div className="app">
        <InputContainer
          jobTypes={this.state.jobTypes}
          data={this.state.data}
          submitRequest={this.sendMsg}
        />
        <HeatMap/>
      </div>
    );
  }
}

ReactDOM.render(<App />, document.getElementById('root'));

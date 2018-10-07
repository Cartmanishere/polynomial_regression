import React, { Component } from 'react';
import './App.css';
import Button from '@material-ui/core/Button';
import * as tf from '@tensorflow/tfjs';
import {generateData} from "./data";
import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';
import TextField from '@material-ui/core/TextField';
import Divider from '@material-ui/core/Divider';

class App extends Component {

    constructor(props){
        super(props);

        this.state = {}
        this.state.console = []
        this.consolePrint = this.consolePrint.bind(this);

        this.state.coeff = {a: 0.2, b: 0.2, c: 0.2};
        this.epochs = 250;
        this.numPoints = 1000;
        this.learningRate = 0.1;

        this.predict = this.predict.bind(this);
        this.loss = this.loss.bind(this);
        this.train = this.train.bind(this);
        this.beginTrain = this.beginTrain.bind(this);
        this.handleChange = this.handleChange.bind(this);
        this.clearConsole = this.clearConsole.bind(this);
    }

    handleChange(event) {
        if (event.target.name === 'epochs') {
            this.epochs = parseInt(event.target.value);
        }
        else{
            const coeff = this.state.coeff;
            coeff[event.target.name] = parseFloat(event.target.value);
            this.setState({coeff: coeff});
        }
    }

    consolePrint(message) {
        const messages = this.state.console;
        messages.push(message);
        this.setState({console: messages});
    }

    loss(predictions, labels) {
        const mse = predictions.sub(labels).square().mean();
        return mse;
    }

    predict(x) {
        return tf.tidy(() => {
            const two = tf.scalar(2)
            return this.a.mul(x.pow(two)).add(this.b.mul(x)).add(this.c);
        });
    }

    train(xs, ys, epochs) {
        const optimizer = tf.train.sgd(this.learningRate);

        for (let i=0; i<epochs; i++) {
            optimizer.minimize(() => {
                const preds = this.predict(xs);
                const loss = this.loss(preds, ys);
                if (i % (epochs / 10) === 0) {
                    this.consolePrint('Epoch: ' + (i + 1) + ' Loss: ' + loss.dataSync())
                }
                return loss;
            });
        }
    }

    clearConsole() {
        const messages = this.state.console;
        while (messages.pop() !== undefined){
            // nothing
        }
        this.setState({console: messages});
    }

    async beginTrain() {
        this.clearConsole();
        this.a = tf.variable(tf.scalar(Math.random()));
        this.b = tf.variable(tf.scalar(Math.random()));
        this.c = tf.variable(tf.scalar(Math.random()));

        this.consolePrint('Tensorflow variables generated.')
        this.consolePrint('Using MSE loss and SGD optimizer.')

        this.trainingData = generateData(this.numPoints, this.state.coeff);

        this.consolePrint(this.numPoints + ' data points generated.');
        this.consolePrint('Initial coefficient values: ['+ this.a.dataSync() +', '+ this.b.dataSync() +', '+ this.c.dataSync() +']')
        this.consolePrint('=== Training starts ===')
        await this.train(this.trainingData.xs, this.trainingData.ys, this.epochs);

        this.consolePrint('=== Training ends ===')
        this.consolePrint('Final coefficient values: ['+ this.a.dataSync() +', '+ this.b.dataSync() +', '+ this.c.dataSync() +']')
    }

    render() {
        return (
            <div className={'body'} style={{'padding': '20px'}}>
                <div className={'controls'}>
                    <Paper style={{padding: 5, paddingLeft: 20}}>
                        <Grid container>
                            <Grid item xs>
                                <TextField
                                    label="A coefficient"
                                    type="number"
                                    name={'a'}
                                    InputLabelProps={{
                                        shrink: true,
                                    }}
                                    margin="normal"
                                    min={0}
                                    step={0.01}
                                    onChange={this.handleChange}
                                />
                            </Grid>
                            <Grid item xs>
                                <TextField
                                    label="B Coefficient"
                                    type="number"
                                    name={'b'}
                                    InputLabelProps={{
                                        shrink: true,
                                    }}
                                    margin="normal"
                                    onChange={this.handleChange}
                                />
                            </Grid>
                            <Grid item xs>
                                <TextField
                                    label="C coefficient"
                                    type="number"
                                    name={'c'}
                                    InputLabelProps={{
                                        shrink: true,
                                    }}
                                    margin="normal"
                                    min={0}
                                    step={0.01}
                                    onChange={this.handleChange}
                                />
                            </Grid>
                            <Grid item xs>
                                <TextField
                                    label="Epochs"
                                    type="number"
                                    name={'epochs'}
                                    InputLabelProps={{
                                        shrink: true,
                                    }}
                                    margin="normal"
                                    onChange={this.handleChange}
                                />
                            </Grid>
                        </Grid>
                        <Button variant="contained" color="primary" style={{marginTop: 10, marginBottom: 10}} onClick={this.beginTrain}>
                            Start Training
                        </Button>
                    </Paper>
                </div>
                <br/>
                <Paper style={{padding: 5, paddingLeft: 20, backgroundColor: "#222", color: "white"}}>
                    <h2 align={'center'}>Tensorflow JS Polynomial Regression</h2>
                    {<h4 align={'center'}>y = {this.state.coeff.a} x<sup>2</sup> + {this.state.coeff.b} x + {this.state.coeff.c}</h4>}

                    <Divider light />

                    {this.state.console.map(value => (
                        <pre key={Math.random()}>&gt;&nbsp;{value}</pre>
                    ))}

                </Paper>
            </div>
        );
    }
}

export default App;

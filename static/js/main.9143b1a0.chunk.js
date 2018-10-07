(window.webpackJsonp=window.webpackJsonp||[]).push([[0],{161:function(e,t,n){e.exports=n(392)},166:function(e,t,n){},170:function(e,t,n){},176:function(e,t){},178:function(e,t){},211:function(e,t){},212:function(e,t){},392:function(e,t,n){"use strict";n.r(t);var a=n(2),i=n.n(a),o=n(24),r=n.n(o),s=(n(166),n(87)),c=n.n(s),l=n(152),h=n(153),u=n(154),m=n(159),d=n(155),f=n(160),b=n(20),p=(n(170),n(157)),g=n.n(p),v=n(19);function y(e,t){var n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:.04;return v.d(function(){var a=[v.c(t.a),v.c(t.b),v.c(t.c)],i=a[0],o=a[1],r=a[2],s=v.b([e],-1,1),c=v.c(2),l=i.mul(s.pow(c)).add(o.mul(s)).add(r).add(v.a([e],0,n));return{xs:s,ys:l}})}var E=n(35),j=n.n(E),k=n(60),C=n.n(k),O=n(46),w=n.n(O),P=n(158),S=n.n(P),x=function(e){function t(e){var n;return Object(h.a)(this,t),(n=Object(m.a)(this,Object(d.a)(t).call(this,e))).state={},n.state.console=[],n.consolePrint=n.consolePrint.bind(Object(b.a)(Object(b.a)(n))),n.state.coeff={a:.2,b:.2,c:.2},n.epochs=100,n.numPoints=1e3,n.learningRate=.1,n.predict=n.predict.bind(Object(b.a)(Object(b.a)(n))),n.loss=n.loss.bind(Object(b.a)(Object(b.a)(n))),n.train=n.train.bind(Object(b.a)(Object(b.a)(n))),n.beginTrain=n.beginTrain.bind(Object(b.a)(Object(b.a)(n))),n.handleChange=n.handleChange.bind(Object(b.a)(Object(b.a)(n))),n.clearConsole=n.clearConsole.bind(Object(b.a)(Object(b.a)(n))),n}return Object(f.a)(t,e),Object(u.a)(t,[{key:"handleChange",value:function(e){if("epochs"===e.target.name)this.epochs=parseInt(e.target.value);else{var t=this.state.coeff;t[e.target.name]=parseFloat(e.target.value),this.setState({coeff:t})}}},{key:"consolePrint",value:function(e){var t=this.state.console;t.push(e),this.setState({console:t})}},{key:"loss",value:function(e,t){return e.sub(t).square().mean()}},{key:"predict",value:function(e){var t=this;return v.d(function(){var n=v.c(2);return t.a.mul(e.pow(n)).add(t.b.mul(e)).add(t.c)})}},{key:"train",value:function(e,t,n){for(var a=this,i=v.e.sgd(this.learningRate),o=function(o){i.minimize(function(){var i=a.predict(e),r=a.loss(i,t);return o%(n/10)===0&&a.consolePrint("Epoch: "+(o+1)+" Loss: "+r.dataSync()),r})},r=0;r<n;r++)o(r)}},{key:"clearConsole",value:function(){for(var e=this.state.console;void 0!==e.pop(););this.setState({console:e})}},{key:"beginTrain",value:function(){var e=Object(l.a)(c.a.mark(function e(){return c.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return this.clearConsole(),this.a=v.f(v.c(Math.random())),this.b=v.f(v.c(Math.random())),this.c=v.f(v.c(Math.random())),this.consolePrint("Tensorflow variables generated."),this.consolePrint("Using MSE loss and SGD optimizer."),this.trainingData=y(this.numPoints,this.state.coeff),this.consolePrint(this.numPoints+" data points generated."),this.consolePrint("Initial coefficient values: ["+this.a.dataSync()+", "+this.b.dataSync()+", "+this.c.dataSync()+"]"),this.consolePrint("=== Training starts ==="),e.next=12,this.train(this.trainingData.xs,this.trainingData.ys,this.epochs);case 12:this.consolePrint("=== Training ends ==="),this.consolePrint("Final coefficient values: ["+this.a.dataSync()+", "+this.b.dataSync()+", "+this.c.dataSync()+"]");case 14:case"end":return e.stop()}},e,this)}));return function(){return e.apply(this,arguments)}}()},{key:"render",value:function(){return i.a.createElement("div",{className:"body",style:{padding:"20px"}},i.a.createElement("div",{className:"controls"},i.a.createElement(C.a,{style:{padding:5,paddingLeft:20}},i.a.createElement(j.a,{container:!0},i.a.createElement(j.a,{item:!0,xs:!0},i.a.createElement(w.a,{label:"A coefficient",type:"number",name:"a",InputLabelProps:{shrink:!0},margin:"normal",min:0,step:.01,onChange:this.handleChange})),i.a.createElement(j.a,{item:!0,xs:!0},i.a.createElement(w.a,{label:"B Coefficient",type:"number",name:"b",InputLabelProps:{shrink:!0},margin:"normal",onChange:this.handleChange})),i.a.createElement(j.a,{item:!0,xs:!0},i.a.createElement(w.a,{label:"C coefficient",type:"number",name:"c",InputLabelProps:{shrink:!0},margin:"normal",min:0,step:.01,onChange:this.handleChange})),i.a.createElement(j.a,{item:!0,xs:!0},i.a.createElement(w.a,{label:"Epochs",type:"number",name:"epochs",InputLabelProps:{shrink:!0},margin:"normal",onChange:this.handleChange}))),i.a.createElement(g.a,{variant:"contained",color:"primary",style:{marginTop:10,marginBottom:10},onClick:this.beginTrain},"Start Training"))),i.a.createElement("br",null),i.a.createElement(C.a,{style:{padding:5,paddingLeft:20,backgroundColor:"#222",color:"white"}},i.a.createElement("h2",{align:"center"},"Tensorflow JS Polynomial Regression"),i.a.createElement("h4",{align:"center"},this.state.coeff.a," x",i.a.createElement("sup",null,"2")," + ",this.state.coeff.b," x + ",this.state.coeff.c," = 0"),i.a.createElement(S.a,{light:!0}),this.state.console.map(function(e){return i.a.createElement("pre",{key:Math.random()},">\xa0",e)})))}}]),t}(a.Component);Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));r.a.render(i.a.createElement(x,null),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then(function(e){e.unregister()})}},[[161,2,1]]]);
//# sourceMappingURL=main.9143b1a0.chunk.js.map
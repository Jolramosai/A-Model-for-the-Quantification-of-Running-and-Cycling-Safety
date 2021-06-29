import React, { Component } from 'react';
import { StyleSheet, View, Image } from 'react-native';
import SplashScreen from 'react-native-splash-screen';
import LinearGradient from 'react-native-linear-gradient';
import Button2 from './Button';
import axios from 'axios';
import Alert from './vendor/Alert';

export default class App extends Component {

	constructor(props) {
		super(props);
		this.state = {
			visible: false,
			title: '',
			message: '',
			status: '',
		};
	}

	_showAlert = (criteria) => {
		if (criteria === '-2') {
			this.setState({
				title: 'Error',
				message: 'Something went wrong..',
				status: 'error-info',
			})
		} else if (criteria === '-1') {
			this.setState({
				title: 'Info',
				message: 'There is no forecast for this time!',
				status: 'info',
			}) 
		} else if (criteria === '0') {
			this.setState({
				title: 'Minimal Risk',
				message: 'Highly recommended to practice sports!',
				status: 'success',
			}) 
		} else if (criteria === '1') {
			this.setState({
				title: 'Low Risk',
				message: 'You can play sports freely!',
				status: 'warning-light',
			}) 
		} else if (criteria === '2') {
			this.setState({
				title: 'Moderate Risk',
				message: 'Practice sport in moderation!',
				status: 'warning',
			}) 
		} else if (criteria === '3') {
			this.setState({
				title: 'High Risk',
				message: 'Sport is not recommended! Stay at home!',
				status: 'error',
			}) 
		}
        this.setState({
            visible: true
        })
	}
	
	_hideAlert = () => {
        this.setState({
            visible: false
        })
	}
	
	_checkPredictions = async () => {
		const today = new Date();
		const day = today.getDate();
		const hour = today.getHours();
		let month = today.getMonth();
		month = month + 1;
		const headers = {
			"Content-Type": "application/x-www-form-urlencoded",
			"Accept": "application/json",
		}
		const url = `http://ec2-18-132-195-71.eu-west-2.compute.amazonaws.com:5000/predict?h=${hour}&d=${day}&m=${month}`
		await axios.get(url, { headers })
			.then((response) => {
				const data = response.data;
				this._showAlert(data.criteria);
			})
			.catch((error) => {
				this._showAlert('-2');
			});
	};

	componentDidMount = async () => {
		console.disableYellowBox = true;
		setTimeout(() => {
			SplashScreen.hide();
		}, 1500);
	};

	render() {
		return (
			<LinearGradient
				colors={[ '#ffffff', '#a6a6a6', '#a6a6a6', '#ffffff' ]}
				style={{ flex: 1,
						 alignItems: 'center',
						 justifyContent: 'center' }}
				>
				<View style={styles.containerUp}>
					<Alert
						visible={this.state.visible}
						title={this.state.title}
						showButton={true}
						message={this.state.message}
						status={this.state.status}
						onConfirmPressed={this._hideAlert}
					/>
				</View>
				<View style={styles.containerDown}>
					<Image
						style={styles.nameLogo}
						resizeMode="contain"
						source={require('./logo.png')}
					/>
				</View>
				<View style={{ flexDirection: 'row', justifyContent: 'space-between', padding: 10, borderRadius: 150, marginBottom: 40 }}>
					<Button2 text="    Check Predictions    " size="default" theme="primary" onPress={this._checkPredictions} />
				</View>
			</LinearGradient>
		);
	}
}

const styles = StyleSheet.create({
	backgroundImage: {
		flex: 1,
		width: '100%',
		height: '100%',
		flexDirection: 'column',
		backgroundColor: 'transparent',
		justifyContent: 'flex-start'
	},
	containerUp: {
		flex: 1,
        alignItems: 'center',
		justifyContent: 'center',
		marginTop: 120
	},
	containerDown: {
		flex: 1,
		flexDirection: 'row',
		justifyContent: 'center',
		marginRight: 10,
		marginTop: 30,
	},
	nameLogo: {
		width: '90%',
		height: '90%'
	},
});

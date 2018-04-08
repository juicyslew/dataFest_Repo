import React from 'react';
import PropTypes from 'prop-types';

export default class Input extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      jobType: null,
      edLevel: 0,
      supervisor: false,
      experience: 0,
    };
  }

  onInputChange = (e) => {
    const value  = e.target.type === 'checkbox' ? e.target.checked : e.target.value;
    this.setState({ [e.target.name]: value });
    console.log(`Name: ${e.target.name} Value: ${value}`);
  };

  submitQuery = () => {
    this.props.submitRequest(this.state);
  };

  render() {
    const jobTypes = [];
    this.props.jobTypes.forEach((job) => {
      jobTypes.push(<option key={job} name={job} value={job}>{job}</option>);
    });

    const edLevels = [];
    this.props.educationLevels.forEach((edLevel) => {
      edLevels.push((
        <label key={edLevel.value}>
          <input
            type="radio"
            name="edLevel"
            value={edLevel.value}
            checked={edLevel.value === this.state.edLevel}
            onChange={this.onInputChange}
          />
          {edLevel.name}
        </label>
      ));
    });

    return (
      <div className="input-container">
        <h1>Applicant Background</h1>
        <div className="education">
          <h2>Education</h2>
          {edLevels}
        </div>
        <h2>Job Type</h2>
        <select name="jobType" onChange={this.onInputChange}>
          {jobTypes}
        </select>
        <h2>Role</h2>
        <label className="supervisor">
          <input type="checkbox" name="supervisor" onChange={this.onInputChange} checked={this.state.supervisor} />
          I am looking for a supervisor position
        </label>
        <h2>Years of Experience</h2>
        <input type="number" name="experience" value={this.state.experience} onChange={this.onInputChange}/>
        <button onClick={this.submitQuery}>Submit</button>
      </div>
    )
  }

}

Input.propTypes = {
  data: PropTypes.object,
  jobTypes: PropTypes.arrayOf(PropTypes.string),
  educationLevels: PropTypes.arrayOf(PropTypes.object),
  submitRequest: PropTypes.func.isRequired,
};

Input.defaultProps = {
  data: null,
  jobTypes: [
    'mednurse',
    'swdev',
  ],
  educationLevels: [
    {name: 'None', value: "0"},
    {name: 'High School', value: "1"},
    {name: 'College', value: "2"},
  ],

};
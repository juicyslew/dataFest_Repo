import React from 'react';
import PropTypes from 'prop-types';
import { Map, TileLayer } from 'react-leaflet';
import HeatmapLayer from 'react-leaflet-heatmap-layer';
import { addressPoints } from './realworld.10000';

export default class HeatMap extends React.Component {

  render() {
    const gradient = {
      0.1: '#89BDE0', 0.2: '#96E3E6', 0.4: '#82CEB6',
      0.6: '#FAF3A5', 0.8: '#F5D98B', '1.0': '#DE9A96'
    };
    return (
      <Map center={this.props.center} zoom={this.props.zoom}>
        <HeatmapLayer
          fitBoundsOnLoad
          fitBoundsOnUpdate
          points={addressPoints}
          gradient={gradient}
          longitudeExtractor={m => m[1]}
          latitudeExtractor={m => m[0]}
          intensityExtractor={m => parseFloat(m[2])}
        />
        <TileLayer
          attribution="&amp;copy <a href=&quot;http://osm.org/copyright&quot;>OpenStreetMap</a> contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
      </Map>
    )
  }

}

HeatMap.propTypes = {
  center: PropTypes.arrayOf(PropTypes.number),
  zoom: PropTypes.number,
};

HeatMap.defaultProps = {
  center: [0, 0],
  zoom: 13,
};
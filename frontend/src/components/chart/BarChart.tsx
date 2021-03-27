import React, { useRef, useLayoutEffect } from 'react';
import * as am4core from '@amcharts/amcharts4/core';
import * as am4charts from '@amcharts/amcharts4/charts';

import am4themes_animated from '@amcharts/amcharts4/themes/animated';
import am4themes_dark from '@amcharts/amcharts4/themes/dark';

am4core.useTheme(am4themes_dark);
am4core.useTheme(am4themes_animated);

function BarChart() {
  useLayoutEffect(() => {
    // create chart
    let chart = am4core.create('bar-chart', am4charts.XYChart);
    chart.padding(40, 40, 40, 40); // 상, 하, 좌, 우

    // Y축
    let categoryAxis = chart.yAxes.push(new am4charts.CategoryAxis());
    categoryAxis.renderer.grid.template.location = 0;
    categoryAxis.dataFields.category = 'network';
    categoryAxis.renderer.minGridDistance = 1;
    categoryAxis.renderer.inversed = true;
    categoryAxis.renderer.grid.template.disabled = true;

    // X축
    let valueAxis = chart.xAxes.push(new am4charts.ValueAxis());
    valueAxis.min = 0;

    // 데이터 연결(차트 유형 별로 관련 시리즈가 따로 존재함)
    let series = chart.series.push(new am4charts.ColumnSeries());
    series.dataFields.categoryY = 'network';
    series.dataFields.valueX = 'MAU';
    series.tooltipText = '{valueX.value}';
    series.columns.template.strokeOpacity = 0;
    series.columns.template.column.cornerRadiusBottomRight = 5;
    series.columns.template.column.cornerRadiusTopRight = 5;

    // 내부 label
    let labelBullet = series.bullets.push(new am4charts.LabelBullet());
    labelBullet.label.horizontalCenter = 'left';
    labelBullet.label.dx = 10; // 라벨 text의 margin
    labelBullet.label.text =
      "{values.valueX.workingValue.formatNumber('#.0as')}"; // 소숫점 자리
    labelBullet.locationX = 1;

    // 정렬
    categoryAxis.sortBySeries = series;

    chart.data = [
      {
        network: 'Javascript',
        MAU: 2255250,
      },
      {
        network: 'ruby',
        MAU: 430000,
      },
      {
        network: 'C++',
        MAU: 1000000,
      },
      {
        network: 'reactjs',
        MAU: 246500,
      },
      {
        network: 'django',
        MAU: 355000,
      },
      {
        network: 'mysql',
        MAU: 500000,
      },
      {
        network: 'ios',
        MAU: 62400,
      },
      {
        network: 'swift',
        MAU: 329500,
      },
      {
        network: 'android',
        MAU: 100000,
      },
      {
        network: 'nodejs',
        MAU: 431000,
      },
      {
        network: 'Python',
        MAU: 1433333,
      },
      {
        network: 'Java',
        MAU: 1900000,
      },
    ];
    return () => {
      // dispose를 안해주면 warning뜹니다.
      chart.dispose();
    };
  }, []);

  return <div id="bar-chart" style={{ height: '500px' }}></div>;
}
export default BarChart;

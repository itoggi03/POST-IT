// eslint-disable import/no-extraneous-dependencies
import React from 'react';
import ReactDOM from 'react-dom';
import 'fullpage.js/vendors/scrolloverflow'; // Optional. When using scrollOverflow:true
import ReactFullpage from '@fullpage/react-fullpage';
import { Link } from 'react-router-dom';
import graph1 from 'assets/images/graph1.png';
import graph2 from 'assets/images/graph2.png';
import graph3 from 'assets/images/graph3.png';
import contents from 'assets/images/contents.png';
import mycontents from 'assets/images/mycontents.png';
import styled from 'styled-components';

const Wrapper = styled.div`
  .go-to-report {
    color: #fff;
    margin-top: 25px;
    display: block;
  }
  .title-1 {
    font-size: 5em;
    text-align: center;
    color: #fff;
    font-weight: 700;
  }
  .title-2 {
    display: block;
    font-size: 3em;
    text-align: center;
    color: #fff;
    font-weight: 700;
  }

  .title-2-1 {
    display: inline;
    font-size: 3em;
    text-align: center;
    color: #fff;
    font-weight: 700;
    vertical-align: middle;
  }

  .sub-title {
    font-size: 1.5em;
    color: #fff;
    font-weight: 300;
    margin-bottom: 10px;
  }

  .sub-content {
    font-size: 1.5em;
    color: #fff;
    font-weight: 300;
    margin-top: 20px;
  }
  .fp-section {
    text-align: center;
  }
`;

const Graph = styled.img`
  width: 350px;
  margin-left: 16px;
  margin-right: 16px;
  margin-top: 40px;
`;

const Contents = styled.img`
  display: inline-block;
  width: 350px;
  margin-left: 16px;
  margin-right: 16px;
  margin-top: 40px;
  vertical-align: middle;
`;

const Button = styled.button`
  padding: 0;
  background: #222222;
  border-radius: 5px;
  border-color: transparent;
  display: block;
  color: #fff;
  margin: 0 auto;
  cursor: pointer;
  font-size: 0.85em;
  margin-top: 30px;
`;

interface IProps {
  state: any;
  fullpageApi: any;
}

class FullpageWrapper extends React.Component {
  // onLeave(origin: any, destination:any, direction:any) {
  //   console.log('Leaving section ' + origin.index);
  // }
  // afterLoad(origin, destination, direction) {
  //   console.log('After load: ' + destination.index);
  // }
  render() {
    return (
      <ReactFullpage
        scrollOverflow={true}
        sectionsColor={['#222222', '#222222', '#222222', '#222222', '#222222']}
        // onLeave={this.onLeave.bind(this)}
        // afterLoad={this.afterLoad.bind(this)}
        render={({ state, fullpageApi }: IProps) => {
          return (
            <Wrapper>
              <div className="section section1">
                <h5 className="sub-title">IT 트렌드를 담다</h5>
                <h3 className="title-1">POST-IT.</h3>
              </div>
              <div className="section">
                <h4 className="title-2">
                  다양한 그래프를 통해 최신 IT 트렌드를 확인해보세요.
                </h4>
                <Graph src={graph1} alt="graph1"></Graph>
                <Graph src={graph2} alt="graph2"></Graph>
                <Graph src={graph3} alt="graph3"></Graph>

                <h6 className="sub-content">
                  분야별 통계를 통해 더욱더 자세한 트렌드를 확인할 수 있어요.
                </h6>
                {/* </div> */}
                {/* <div className="slide">
                  <h3>Slide 2.2</h3>
                </div>
                <div className="slide">
                  <h3>Slide 2.3</h3>
                </div> */}
              </div>
              <div className="section">
                <Contents src={contents} alt="graph3"></Contents>
                <h4 className="title-2-1">
                  최신 트렌드 관련 콘텐츠를 확인해보세요.
                </h4>
              </div>
              <div className="section">
                <h4 className="title-2-1">
                  다시 보고 싶은 콘텐츠를 스크랩하여 모아보세요.
                </h4>
                <Contents src={mycontents} alt="mycontents"></Contents>
              </div>
              <div className="section">
                <h4 className="title-2">
                  이제 POST-IT과 함께 개발자로 성장해보세요!
                </h4>
                <Link to="/" className="go-to-report">
                  IT트렌드 확인하러 가기!
                </Link>
                <Button onClick={() => fullpageApi.moveTo(1, 0)}>맨위로</Button>
              </div>
            </Wrapper>
          );
        }}
      />
    );
  }
}

// ReactDOM.render(<FullpageWrapper />, document.getElementById('react-root'));

export default FullpageWrapper;

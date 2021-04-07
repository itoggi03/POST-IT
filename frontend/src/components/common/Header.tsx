import React, { useState, useEffect } from 'react';
import { NavLink, NavLinkProps } from 'react-router-dom';

import { getCurrentUser } from 'api/user';

import Login from 'pages/Login';
import theme from 'assets/theme';
import { Modal } from './Modal';
import { Wrapper, Button } from './Header.styles';
import { PropsTypes, MenuTypes } from 'types/common/headerTypes';
import axios from 'axios';

const MenuItem = ({ to, item, children }: MenuTypes) => (
  <NavLink
    to={to}
    className={`header-${item}`}
    activeStyle={{ color: theme.colors.text.first }}
    isActive={(match) => {
      if (!match) {
        return false;
      }
      return match.isExact;
    }}
    onClick={() => window.scrollTo(0, 0)}
  >
    {children}
  </NavLink>
);

function Header(props: PropsTypes) {
  // console.log(props);
  const [showModal, setShowModal] = useState(false);
  const openModal = () => {
    setShowModal((prev) => !prev);
  };

  return (
    <Wrapper>
      <div>
        <MenuItem to={'/'} item={'logo'} onClick={() => window.scrollTo(0, 0)}>
          POST-IT
        </MenuItem>
        <MenuItem to={'/report'} item={'menus'}>
          IT 보고서
        </MenuItem>
        <MenuItem to={'/contents'} item={'menus'}>
          일일 컨텐츠
        </MenuItem>
        <MenuItem to={'/profile'} item={'menus'}>
          프로필
        </MenuItem>
        {props.authenticated ? (
          <MenuItem to={'/myfolder'} item={'menus'}>
            내 스크랩
          </MenuItem>
        ) : null}
      </div>
      <button
        onClick={() => {
          // axios.get('http://j4c103.p.ssafy.io:5555/api/auth/refresh%27,%7BwithCredentials:true%7D).then((res)=%3E%7B
          axios
            .get('http://j4c103.p.ssafy.io:8443/refresh', {
              withCredentials: true,
            })
            .then((res) => {
              console.log(res.data);
            })
            .catch((err) => {
              console.log(err);
            });
        }}
      >
        토큰 리프레시
      </button>
      <button
        onClick={() => {
          getCurrentUser()
            .then((res) => console.log(res))
            .catch((err) => console.log(err));
        }}
      >
        getUser
      </button>
      {props.authenticated ? (
        <div>
          <Button onClick={props.onLogout}>
            <span>로그아웃</span>
          </Button>
        </div>
      ) : (
        <div>
          <Button onClick={openModal}>로그인</Button>
        </div>
      )}
      <Modal
        showModal={showModal}
        setShowModal={setShowModal}
        children={<Login />}
      ></Modal>
    </Wrapper>
  );
}

export default Header;

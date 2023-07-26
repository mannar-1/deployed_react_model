import React from 'react';

const Navbar = () => {
  const navbarStyle = {
    backgroundColor: 'purple',
    color: 'black',
    margin: '0',
    padding: '1rem', // Add padding if needed
    height:'50px'
  };

  return (
    <nav style={navbarStyle}>
      <h1>KnowUrFood</h1>
    </nav>
  );
};

export default Navbar;

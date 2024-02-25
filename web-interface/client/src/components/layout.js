import React, {useState} from "react"

const socialMediaItems = [
  {
    iconName: "fa-github",
    link: "https://github.com/ben-hayes/one-hand-clapping",
  },
];

const Layout = ({children}) => {
    const [showMobileMenu,setShowMobileMenu] = useState(false);

    const navBar = (
      <nav className="navbar is-primary is-fixed-top" role="navigation" aria-label="main navigation">
        <div className="navbar-brand">
         <a
            role="button"
            className={`navbar-burger ${
              showMobileMenu ? "is-active" : undefined
            }`}
            aria-label="menu"
            aria-expanded="false"
            data-target="navbarBasicExample"
            onClick={() => setShowMobileMenu(!showMobileMenu)}
          >
            <span aria-hidden="true"></span>
            <span aria-hidden="true"></span>
            <span aria-hidden="true"></span>
          </a>
        </div>
        <div
          className={`navbar-end ${
            !showMobileMenu ? "is-hidden-touch" : undefined
          }`}
        >
          {socialMediaItems.map((item) => (
            <React.Fragment key={`navbar-unit-${item.name}`}>
              <link className="navbar-item" to={item.link}>
                {item.name}
              </link>
              <span className="navitem-divider"></span>
            </React.Fragment>
          ))}
        </div>
      </nav>
    );

    const footer = (
      <footer className="footer">
        <div className="columns">
          <div className="column is-one-third">
            <p className="has-text-left is-size-7">
              Created as part of the 2024 Timbre Tools Hackathon 
              <br/>
            </p>
          </div>
        </div>
      </footer>
    );

    return (

      <>
          {navBar}
          {/* Breadcrumbs and children*/}
          <main className="container is-fullhd">
            {children}
          </main>
          {footer}
      </>
    );
}



export default Layout
import { Link, useLocation } from 'react-router-dom'

function Navbar() {
  const location = useLocation()

  const isActive = (path) => location.pathname === path ? 'active' : ''

  return (
    <nav className="navbar">
      <div className="container">
        <div className="navbar-content">
          <Link to="/" className="navbar-logo">
            ⭐ Rating Predictor
          </Link>
          <ul className="navbar-links">
            <li>
              <Link to="/" className={isActive('/')}>Home</Link>
            </li>
            <li>
              <Link to="/predict" className={isActive('/predict')}>Predict</Link>
            </li>
            <li>
              <Link to="/about" className={isActive('/about')}>About</Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
